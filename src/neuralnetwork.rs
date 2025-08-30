use ndarray::{Array1, Array2};
use rand::Rng;

pub struct LinearLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl LinearLayer{
    pub fn new(input_size: usize, output_size: usize)->LinearLayer{
        let mut rng = rand::rng();

        // He initialization (good for ReLU)
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.random_range(-1.0..1.0) * (2.0 / input_size as f32).sqrt()
        });

        let biases = Array1::zeros(output_size);

        Self { weights, biases }
    }
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.biases
    }
}

pub struct NeuralNetwork {
    layers: Vec<LinearLayer>,
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

    // Optional: clone weights into a new network
    pub fn clone_from(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| LinearLayer {
                weights: layer.weights.clone(),
                biases: layer.biases.clone(),
            }).collect(),
            activation: self.activation,
        }
    }
}
