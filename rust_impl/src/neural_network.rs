use std::collections::HashMap;
use std::fs::File;

use ndarray::{Array2, Axis};
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use ndarray_stats::QuantileExt;

enum ActivationFunction {
    RELU,
    SIGMOID,
    SOFTMAX,
    ARGMAX
}

fn sigmoid_activation(z: Array2<f32>) -> Array2<f32> {
    z.mapv(|x| sigmoid(&x))
}

fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-z))
}

fn relu_activation(z: Array2<f32>) -> Array2<f32> {
    z.mapv(|x| relu(&x))
}

fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

fn softmax_activation(x: Array2<f32>) -> Array2<f32> {
    let sum_exp = x.mapv(|i| i.exp()).sum();
    x.mapv(|i| i.exp() / sum_exp)
}

fn argmax(x: Array2<f32>) -> Array2<f32> {
    let arr: Vec<f32> = x.axis_iter(Axis(1))
        .map(|x| x.argmax().unwrap() as f32)
        .collect();
    Array2::<f32>::from_shape_vec((1, arr.len()), arr).unwrap()
}

enum Layer {
    INPUT(usize),
    HIDDEN(usize, ActivationFunction),
    OUTPUT(usize, ActivationFunction)
}

pub struct NeuralNetwork{
    layers: HashMap<usize, Layer>,
    weights: HashMap<usize, Array2<f32>>,
    biases: HashMap<usize, Array2<f32>>
}

impl NeuralNetwork {
    pub fn init() -> NeuralNetwork {
        // TODO: Dynamic reading!
        let mut layers = HashMap::<usize, Layer>::new();
        layers.insert(0, Layer::INPUT(784));
        layers.insert(1, Layer::HIDDEN(128, ActivationFunction::RELU));
        layers.insert(2, Layer::OUTPUT(10, ActivationFunction::ARGMAX));

        let file = File::open("../feasibility/weights_1.txt").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
        let weights_1: Array2<f32> = reader.deserialize_array2((784, 128)).unwrap();

        let file = File::open("../feasibility/weights_2.txt").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
        let weights_2: Array2<f32> = reader.deserialize_array2((128, 10)).unwrap();

        let file = File::open("../feasibility/biases_1.txt").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
        let biases_1: Array2<f32> = reader.deserialize_array2((128, 1)).unwrap();

        let file = File::open("../feasibility/biases_2.txt").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
        let biases_2: Array2<f32> = reader.deserialize_array2((10, 1)).unwrap();

        let mut weights: HashMap<usize, Array2<f32>> = HashMap::new();
        weights.insert(1, weights_1);
        weights.insert(2, weights_2);

        let mut biases: HashMap<usize, Array2<f32>> = HashMap::new();
        biases.insert(1, biases_1);
        biases.insert(2, biases_2);

        NeuralNetwork {
            layers,
            weights,
            biases
        }
    }

    pub fn predict(&self, input: Array2<f32>) -> Array2<f32> {
        // Forward propagation
        let mut x: Array2<f32> = input;

        let w = self.weights.get(&1).unwrap();
        let b = self.biases.get(&1).unwrap(); 
        let z = w.t().dot(&x) + b;
        x = relu_activation(z);
        
        let w = self.weights.get(&2).unwrap();
        let b = self.biases.get(&2).unwrap(); 
        let z = w.t().dot(&x) + b;
        x = softmax_activation(z);

        return argmax(x);

        // for (idx, layer) in self.layers.iter() {
        //     x = match layer {
        //             Layer::INPUT(idx) => *input,
        //             Layer::HIDDEN(idx, act_fnc) => {
        //                 let w = self.weights.get(idx).unwrap();
        //                 let b = self.biases.get(idx).unwrap();

        //                 // Do the math
        //                 let z = w.dot(&x) * b;
        //                 // Apply activation function
        //                 match act_fnc {
        //                     ActivationFunction::RELU => relu_activation(z),
        //                     ActivationFunction::SIGMOID => sigmoid_activation(z),
        //                     ActivationFunction::SOFTMAX => relu_activation(z),
        //                 }
        //             },
        //             Layer::OUTPUT(idx, act_fnc) => {
        //                 let w = self.weights.get(idx).unwrap();
        //                 let b = self.biases.get(idx).unwrap();

        //                 // Do the math
        //                 let z = w.dot(&x) * b;
        //                 // Apply activation function
        //                 match act_fnc {
        //                     ActivationFunction::RELU => relu_activation(z),
        //                     ActivationFunction::SIGMOID => sigmoid_activation(z),
        //                     ActivationFunction::SOFTMAX => relu_activation(z),
        //                 }
        //             },
        //     };
        // }
    }
}
