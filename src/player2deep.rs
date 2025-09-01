use rand::{rng, Rng};

use crate::connect4::Tile;
use crate::deepqlearn::DeepQLearn;
use crate::neuralnetwork::NeuralNetwork;
use crate::qlearn::QLearn;
pub struct Player2Deep{
    pub qnet: NeuralNetwork,
    pub color: Tile
}

impl Player2Deep{
    pub fn new(qnet: NeuralNetwork, color:Tile)->Player2Deep{
        Player2Deep{qnet, color}
    }
    pub fn turn(&self, state: &mut DeepQLearn){
        /* if let Some(actions) = self.qnet.forward(&state.state){ */
        let mut rng = rng();
        let actions = self.qnet.forward(&state.state);
        let (_, indices) = actions.iter().enumerate().fold((f32::NEG_INFINITY, Vec::new()), |(mx, mut indices), (idx, &val)|{if (mx - val).abs() < 1e-6 {indices.push(idx); (mx, indices)} else if val < mx {(mx, indices)} else {(val, vec![idx])}});
        let action = indices[rng.random_range(0..indices.len())];
        state.insert(action, self.color.clone());

        /* }else{
            let mut rng = rng();
            state.insert(rng.random_range(0..state.cols), self.color.clone());
        } */
    }

    pub fn self_move(&self, col:usize, state: &mut DeepQLearn){
        state.insert(col, self.color.clone());
    }
}