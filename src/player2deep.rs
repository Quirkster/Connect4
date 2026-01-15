use rand::{rng, Rng};
use rand_distr::{Distribution, StandardNormal};
use rand_distr::weighted::WeightedIndex;

use crate::alphabeta;
use crate::connect4::Tile;
use crate::deepqlearn::{DeepQLearn, ReplayTuple};
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
    pub fn turn(&self, state: &mut DeepQLearn, random: bool)->bool{
        /* if let Some(actions) = self.qnet.forward(&state.state){ */
        let player_num = match self.color{
            Tile::Red => 1.0,
            Tile::Blue => -1.0,
            Tile::Empty => 0.0,
        };
        let actions = self.qnet.forward(&(player_num * &state.state));
        //println!("{:?}", actions);
        let action;
        if !random{
            let mut rng = rand::rng();
            let (_, indices) = actions.iter().enumerate().filter(|(idx, _)| state.is_action_valid(*idx)).fold((f32::NEG_INFINITY, Vec::new()), |(mx, mut indices), (idx, &val)|{if (mx - val).abs() < 1e-6 {indices.push(idx); (mx, indices)} else if val < mx {(mx, indices)} else {(val, vec![idx])}});
            if indices.len() == 0{
                return true
            }
            action = indices[rng.random_range(0..indices.len())];
        }
        else{
            let dist = WeightedIndex::new(&actions.mapv_into(NeuralNetwork::sigmoid)).unwrap();
            let mut rng = rand::rng();
            action = dist.sample(&mut rng);

        }
        
        let prev_state = state.state.clone();
        let inserted = state.insert(action, self.color.clone());
        if (state.calculate_reward(inserted as usize, action, player_num as i32) - player_num).abs() < 1e-6{
            state.replay_memory.push(ReplayTuple::new(prev_state, 0, -1.0, state.state.clone(), true));
            return true
        }
        /* }else{
            let mut rng = rng();
            state.insert(rng.random_range(0..state.cols), self.color.clone());
        } */
       false
    }

    pub fn self_move(&self, col:usize, state: &mut DeepQLearn)->alphabeta::Result{
        let inserted = state.insert(col, self.color.clone());
        let player_num = match self.color{
            Tile::Red => 1.0,
            Tile::Blue => -1.0,
            Tile::Empty => 0.0,
        };
        let res = state.calculate_reward(inserted as usize, col, player_num as i32);
        if ( res- player_num).abs() < 1e-6{
            return alphabeta::Result::WIN
        }
        if ( res- 0.5).abs() < 1e-6{
            return alphabeta::Result::DRAW
        }
        return alphabeta::Result::INPROGRESS
    }

    pub fn randomize_weights(&mut self){
        let mut rng = rand::rng();
        self.qnet.layers.iter_mut().for_each(|layer| {
            layer.weights.mapv_inplace(|_| rng.sample(StandardNormal));
        });
        
        
    }

}