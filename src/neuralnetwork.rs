use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use std::f32::consts::E;
use std::fs::File;
use std::io::{BufWriter, Write};
use byteorder::{WriteBytesExt, LittleEndian};

use crate::qlearn::EPSILON_DECAY;

pub const WEIGHT_DECAY_L2:f32 = 1e-5;

#[derive(Debug)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub weight_grads: Array2<f32>, 
    pub bias_grads: Array1<f32>,
    
}

impl LinearLayer{
    pub fn new(input_size: usize, output_size: usize)->LinearLayer{
        let mut rng = rand::rng();

        // He initialization (good for ReLU)
        let std = (2.0 / input_size as f32).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.sample::<f32, _>(rand_distr::StandardNormal) * std
        });

        let biases = Array1::zeros(output_size);

        Self { weights, biases, weight_grads: Array2::zeros((output_size, input_size)), bias_grads: Array1::zeros(output_size),}
    }
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.biases
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<LinearLayer>,
    activation: fn(f32) -> f32,
    activation_derivative: fn(f32) -> f32
}

impl NeuralNetwork{
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend_from_slice(hidden_sizes);
        layer_sizes.push(output_size);

        for i in 0..(layer_sizes.len() - 1) {
            layers.push(LinearLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        Self {
            layers,
            activation: |x|{if x > 0.0 { x } else { 0.0 }}, //relu
            activation_derivative: |x| {if x > 0.0 {1.0} else {0.0}}
        
        }
    }

    pub fn from_layers(layers:Vec<LinearLayer>)->Self{
        Self{layers, activation: |x|{x.max(0.0)}, activation_derivative: |x| {if x > 0.0 {1.0} else {0.0}}}
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            // Don't apply activation after last layer
            if i != self.layers.len() - 1 {
                x.mapv_inplace(self.activation);
            }
        }

        x
    }

    pub fn forward_with_cache(&self, input: &Array1<f32>)-> (Vec<Array1<f32>>, Vec<Array1<f32>>){

        assert!(
            !input.iter().any(|x| x.is_nan()),
            "NaN found in input"
        );

        let mut activations = vec![input.clone()];
        //let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut x = input.clone();
        let mut masks = Vec::with_capacity(self.layers.len());
        for (i, mut layer) in self.layers.iter().enumerate(){
            let z = layer.forward(&x);
            //pre_activations.push(z.clone());
            if i != self.layers.len() - 1{
                x = z.mapv(self.activation);

                masks.push(z.mapv((self.activation_derivative)));
            }else{
                //are we sure that we wan't the relu to be all ones
                x = z.clone();
                masks.push(Array1::ones(z.len()));
            }
            //Dropout --> find a better solution;
            //x.iter_mut().for_each(|a| {if *a > 150.0{*a = 0.0}});

            assert!(x.iter().all(|a| a.is_finite()), "NaN in activations");

            activations.push(x.clone());
        }
        (activations, /* pre_activations, */ masks)
    } 

    pub fn backward_and_update(
        &mut self,
        activations: &[Array1<f32>],
        /* pre_activations: &[Array1<f32>], */
        mask: &[Array1<f32>],
        action: usize,
        target: f32,
        learning_rate: f32,
    ) {
        let output = activations.last().unwrap();
        let predicted_q = output[action];
        
        // 1. Compute initial loss gradient: dL/dQ
        let mut delta = Array1::<f32>::zeros(output.len());
        //why 
        //delta[action] = 2.0 * (predicted_q - target);
        delta[action] = predicted_q - target;// * (self.activation_derivative)(target);


        // 2. Backpropagate through layers
        for i in (0..self.layers.len()).rev() {
            let a_prev = &activations[i];
            let m: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> = &mask[i];


            // Activation gradient (always apply unless your output layer uses identity)

            let is_output = i == self.layers.len() - 1;

            //layer delta
            let dz = if is_output {
                delta.clone() // identity activation -> derivative = 1
            } else {
                m * &delta
            };
            
            let layer:&mut LinearLayer = &mut self.layers[i];

                        //println!("{:?}, {m:?}", layer.weights);


            // Gradient for weights and biases
            let grad_w = dz.view().insert_axis(Axis(1))
                .dot(&a_prev.view().insert_axis(Axis(0)))
                + ((WEIGHT_DECAY_L2 )as f32 * &layer.weights);

                /*hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

    self.weights_hidden_output += np.dot(self.hidden_output.T,
                                         output_delta) * learning_rate
    self.bias_output += np.sum(output_delta, axis=0,
                               keepdims=True) * learning_rate
    self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    self.bias_hidden += np.sum(hidden_delta, axis=0,
                               keepdims=True) * learning_rate */

            //layer.bias_grads = grad_w.sum_axis(Axis(1));

            layer.weight_grads = grad_w;
            layer.bias_grads = dz.clone();

            let delta_prev = layer.weights.t().dot(&dz);
            // Update weights and biases

            //println!("{:?}", layer.weight_grads.fold(0, |mut acc, &x|{if x > 0.05{acc += 1;} acc}));
            layer.weights -= &(learning_rate * &layer.weight_grads);

            layer.biases -= &(learning_rate * &layer.bias_grads);

            // Propagate delta to the previous layer
            delta = delta_prev;
        }
    }

    pub fn backward_accumulate(
        &mut self,
        activations: &[Array1<f32>],
        /* pre_activations: &[Array1<f32>], */
        mask: &[Array1<f32>],
        action: usize,
        target: f32,
        dw_acc:&mut [Array2<f32>],
        db_acc: &mut [Array1<f32>],


    ) {
        let output = activations.last().unwrap();
        let predicted = output[action];

        //For MSE loss: (1/2)*(pred-target)^2  => dL/dpred = (pred-target)
        //If you use plain (pred-target)^2, derivative is 2*(pred-target).
        let dl_dq = predicted - target;

        // delta is gradient wrt pre-activation at current layer (dL/dz)
        let mut delta = Array1::<f32>::zeros(output.len()); 
        delta[action] = dl_dq;                // sparse TD error on chosen action

        

        // Backprop through layers from last to first
        for i in (0..self.layers.len()).rev(){
            let is_output = i == self.layers.len() - 1;
            let a_prev =&activations[i];
            let m = &mask[i];
            let dz = if is_output {
                delta.clone() // identity activation -> derivative = 1
            } else {
                m * &delta
            };

            let layer:&mut LinearLayer = &mut self.layers[i];
            let dw = dz.view().insert_axis(Axis(1))
                        .dot(&a_prev.view().insert_axis(Axis(0))) + ((WEIGHT_DECAY_L2 )as f32 * &layer.weights);

            dw_acc[i] += &dw;
            db_acc[i] += &dz;
            delta = (layer.weights).t().dot( &dz);



        }
  
  
    }

    pub fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn softmax(input: &Array1<f32>)-> Array1<f32>{
        let e_i = input.map(|&i|{E.powf(i)});
        let total_e_i = e_i.sum();
        e_i.map(|&x|{x/total_e_i})
    }

    pub fn sigmoid(input: f32)->f32{
        1.0/(1.0 + E.powf(-input))
    }

    // Optional: clone weights into a new network
    pub fn clone_from(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| LinearLayer {
                weights: layer.weights.clone(),
                biases: layer.biases.clone(),
                weight_grads: layer.weight_grads.clone(),
                bias_grads: layer.bias_grads.clone()
            }).collect(),
            activation: self.activation,
            activation_derivative: self.activation_derivative
        }
    }
}
