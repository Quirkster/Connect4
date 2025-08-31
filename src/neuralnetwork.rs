use ndarray::{Array1, Array2};
use rand::Rng;

use std::fs::File;
use std::io::{BufWriter, Write};
use byteorder::{WriteBytesExt, LittleEndian};

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
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.random_range(-1.0..1.0) * (2.0 / input_size as f32).sqrt()
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
            activation: |x|{x.max(0.0)} //relu
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
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate(){
            let z = layer.forward(&x);
            pre_activations.push(z.clone());

            if i != self.layers.len() - 1{
                x = z.mapv(self.activation);
            }else{
                x = z;
            }

            activations.push(x.clone());
        }
        (activations, pre_activations)
    }

    pub fn backward_and_update(
        &mut self,
        activations: &[Array1<f32>],
        pre_activations: &[Array1<f32>],
        action: usize,
        target: f32,
        learning_rate: f32,
    ) {
        let output = activations.last().unwrap();
        let mut delta = Array1::<f32>::zeros(output.len());
        let predicted_q = output[action];

        delta[action] = 2.0 * (predicted_q - target);

        for i in (0..self.layers.len()).rev(){
            let a_prev = &activations[i];
            let z = &pre_activations[i];

            let dz = if i != self.layers.len() - 1 {
                z.mapv(|v| Self::relu_derivative(v)) * &delta
            } else {
                delta.clone()
            };

            let layer = &mut self.layers[i];

            // Gradient for weights and biases
            let grad_w = dz.view().insert_axis(ndarray::Axis(1))
                .dot(&a_prev.view().insert_axis(ndarray::Axis(0)));

            layer.weight_grads = grad_w;
            layer.bias_grads = dz.clone();

            // Propagate delta to previous layer
            delta = layer.weights.t().dot(&dz);

            // Gradient descent update
            layer.weights -= &(learning_rate * &layer.weight_grads);
            layer.biases -= &(learning_rate * &layer.bias_grads);

        }
    }

    pub fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
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
