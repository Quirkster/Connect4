use ndarray::{array, Array, Array1};
use rand::{seq::IndexedRandom, Rng};
use rand::seq::SliceRandom;

use crate::{connect4::Tile, qlearn::{ALPHA, EPSILON_DECAY, EPSILON_MIN, GAMMA}};
use crate::neuralnetwork::NeuralNetwork;

pub const BATCH_SIZE:usize = 6;
pub struct ReplayTuple{
    state: Array1<f32>,
    action: usize,
    reward: f32,
    next_state: Array1<f32>,
    done:bool
}
impl ReplayTuple{
    pub fn new(state: Array1<f32>, action: usize, reward:f32, next_state: Array1<f32>, done: bool)->ReplayTuple{
        ReplayTuple { state, action, reward, next_state, done}
    }
}
pub struct DeepQLearn{
    pub replay_memory: Vec<ReplayTuple>,
    pub action_value: NeuralNetwork,
    pub target: NeuralNetwork,
    pub epsilon: f64,
    pub state: Array1<f32>,
    pub rows: usize,
    pub cols:usize,
    pub player: i32,
}

impl DeepQLearn{
    pub fn new(rows:usize, cols:usize, player: i32)->DeepQLearn{
        let action_value = NeuralNetwork::new(42, &[64, 64], 7);
        let target = NeuralNetwork::clone_from(&action_value);
        DeepQLearn{replay_memory:Vec::new(), action_value, target, epsilon:1.0,  state: Array::zeros([rows*cols;1]), rows, cols, player}
    }
    pub fn insert(&mut self, col:usize, color: Tile)->bool{
        for row in (0..self.rows).rev(){
            if (self.state[row * self.cols + col] - 0.0).abs() < 1e-6{
                self.state[row * self.cols + col] = match color {
                    Tile::Red => 1.0,
                    Tile::Blue => 2.0,
                    Tile::Empty => 0.0,
                };
                println!("inserted into col: {col}");
                return true
            }
        }
        false
    }
    pub fn clear_board(&mut self){
        (0..(self.rows*self.cols)).for_each(|i|{
            self.state[i] = 0.0;
        })
    }
    fn is_action_valid(&self, col:usize)->bool{
        match (0..self.rows).rev().find(|&el| (self.state[el* self.cols + col] - 0.0).abs() < 1e-6){
            Some(_) => true,
            None => false
        }
    }
    ///calculate reward for 4x4 
/// returns 1 if victory, -1 if loss, 0 otherwise
    pub fn calculate_reward(&self)->f32{
        let (rows, cols) = (self.rows, self.cols);

        let directions: [(i32, i32); 4] = [
            (0, 1),   
            (1, 0),  
            (1, 1),  
            (1, -1),
        ];

        for r in 0..rows{
            for c in 0..cols{
                if (self.state[r * cols + c] - 0.0) < 1e-6{
                    continue
                }
                let player = self.state[r * cols + c];
                if let Some(_) = directions.iter().find(|(dr, dc)|{
                    let mut count = 0;
                    for i in 0..4 {
                        let nr = r as i32 + dr * i;
                        let nc = c as i32 + dc * i;
                        if 0 <= nr && (nr as usize) < rows && 0 <= nc && (nc as usize) < cols && self.state[nr as usize * cols + nc as usize] == player{
                            count += 1
                        }else{
                            break
                        }
                    }
                    if count == 4{
                        return true
                    }
                    false
                }){
                    return player
                }
            }
        }

        return 0.0
    }
}

impl Iterator for DeepQLearn{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = rand::rng();
        let rand = rng.random_range(0.0..1.0);
        let prev_state = self.state.clone();
        let mut action;
        if rand < self.epsilon {
            action = rng.random_range(0..self.cols);
            
            let mut inserted = self.insert(action, if self.player == 1{Tile::Red} else{Tile::Blue});
            let mut tries = 1;
            while !inserted && tries <= 4{
                action = rng.random_range(0..self.cols);
                inserted = self.insert(action, if self.player == 1{Tile::Red} else{Tile::Blue});
                tries += 1;
            }
            
        }
        else{
            //may have to check that state is initalized
            let actions = self.action_value.forward(&self.state);
            
            action = *actions.iter().max_by(|&&a,&&b|{if (a-b).abs() < 1e-6{return std::cmp::Ordering::Equal} else if a < b {return std::cmp::Ordering::Less} return std::cmp::Ordering::Greater}).expect("invalid action") as usize;
            let mut inserted = self.insert( action, if self.player == 1{Tile::Red} else{Tile::Blue});
            
        }
        let reward = if self.calculate_reward() == (self.player as f32){
            1.0
        }else if self.calculate_reward() == 0.0{
            0.0
        }else{
            -1.0
        };
        println!("{reward}");
        self.replay_memory.push(ReplayTuple::new(prev_state.clone(), action, reward, self.state.clone(), (reward - 0.0).abs() > 1e-6 || (reward + 1.0).abs() < 1e-6));
        let next_q = self.action_value.forward(&self.state);
        let max_q_next = next_q.iter()
            .enumerate()
            .filter(|(a, _)| self.is_action_valid(*a))
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let target = reward + GAMMA as f32 * max_q_next * (1.0 - ((reward - 0.0).abs() > 1e-6) as u8 as f32);
            
        /* let (activations, pre_activations) = self.action_value.forward(&(Array1::from(self.state)));
        self.action_value.backward_and_update(&activations, &pre_activations, action, target, learning_rate);

        state = next_state;
        
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_MIN); */
        //TODO: represent actual reward
        return Some(0.0)
    }
    
}