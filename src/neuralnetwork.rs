use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use std::fs::File;
use std::io::{BufWriter, Write};
use byteorder::{WriteBytesExt, LittleEndian};

use crate::qlearn::EPSILON_DECAY;

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
            activation: |x|{if x > 0.0 { x } else { 0.01 * x }} //relu
        }
    }
    pub fn from_layers(layers:Vec<LinearLayer>)->Self{
        Self{layers, activation: |x|{x.max(0.0)}}
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
        for (i, layer) in self.layers.iter().enumerate(){
            let z = layer.forward(&x);

            //pre_activations.push(z.clone());
            //println!("Layer {}: z = {:?}", i, z);
            if i != self.layers.len() - 1{
                x = z.mapv(self.activation);

                masks.push(z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }));
            }else{
                x = z.clone();
                masks.push(Array1::ones(z.len()));
            }

            assert!(x.iter().all(|a| a.is_finite()), "NaN in activations");
            //println!("Layer {}: a = {:?}", i, x);

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
        delta[action] = 2.0 * (predicted_q - target);

        // 2. Backpropagate through layers
        for i in (0..self.layers.len()).rev() {
            let a_prev = &activations[i];
            let m = &mask[i];
            // Activation gradient (always apply unless your output layer uses identity)
            let is_output = i == self.layers.len() - 1;
            let dz = if is_output {
                delta.clone() // identity activation -> derivative = 1
            } else {
                m * &delta
            };
            
            let layer = &mut self.layers[i];

            // Gradient for weights and biases
            let grad_w = dz.view().insert_axis(Axis(1))
                .dot(&a_prev.view().insert_axis(Axis(0)))
                + (EPSILON_DECAY as f32 * &layer.weights);

            //println!("{dz:?}");
            layer.weight_grads = grad_w;
            layer.bias_grads = dz.clone();

            let delta_prev = layer.weights.t().dot(&dz);
            // Update weights and biases
            layer.weights -= &(learning_rate * &layer.weight_grads);
            layer.biases -= &(learning_rate * &layer.bias_grads);

            // Propagate delta to the previous layer
            delta = delta_prev;
        }
    }

    pub fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.01 }
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
        }
    }
}
