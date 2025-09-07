use ndarray::{ Array, Array1};
use rand::SeedableRng;
use rand::{seq::IndexedRandom, Rng};
use rand::rngs::SmallRng;
use rand::seq::{IteratorRandom, SliceRandom};
use serde::de::value::I32Deserializer;

use crate::{connect4::Tile, qlearn::{ALPHA, EPSILON_DECAY, EPSILON_MIN, GAMMA}};
use crate::neuralnetwork::NeuralNetwork;

pub const BATCH_SIZE: usize = 64;
pub const LEARNING_RATE:f32 = 2e-4;
pub const TARGET_UPDATE_FREQ:i32 = 100;
pub const WARMUP:usize = 400;
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
    
    fn get_mirror(rows: usize, cols: usize, prev_state: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>, action: usize, reward: f32, state: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>, done: bool) -> ReplayTuple {
        let mirrored_prev = Array1::from_shape_fn(rows * cols,|j|{let prev_row = j/cols; let prev_col = j % cols; prev_state[prev_row * cols + ((prev_col as i32)-6).abs() as usize]});
        let mirrored_state = Array1::from_shape_fn(rows * cols,|j|{let prev_row = j/cols; let prev_col = j % cols; state[prev_row * cols + ((prev_col as i32)-6).abs() as usize]});
        ReplayTuple{state: mirrored_prev, action: (6-(action as i32)).abs() as usize, reward, next_state: mirrored_state, done}
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
    pub steps: usize
}

impl DeepQLearn{
    pub fn new(rows:usize, cols:usize, player: i32)->DeepQLearn{
        let action_value = NeuralNetwork::new(42, &[64, 64], 7);
        let target = NeuralNetwork::clone_from(&action_value);
        DeepQLearn{replay_memory:Vec::new(), action_value,  target,  epsilon:1.0,  state: Array::zeros([rows*cols;1]), rows, cols, player, steps:0}
    }
    pub fn insert(&mut self, col:usize, color: Tile)->i32{
        for row in (0..self.rows).rev(){
            if (self.state[row * self.cols + col] - 0.0).abs() < 1e-6{
                self.state[row * self.cols + col] = match color {
                    Tile::Red => 1.0,
                    Tile::Blue => 2.0,
                    Tile::Empty => 0.0,
                };
                println!("inserted into col: {col}");
                return row as i32
            }
        }
        -1
    }
    pub fn clear_board(&mut self){
        self.steps = 0;
        (0..(self.rows*self.cols)).for_each(|i|{
            self.state[i] = 0.0;
        })
    }
    pub fn is_action_valid(&self, col:usize)->bool{
        match (0..self.rows).rev().find(|&el| (self.state[el* self.cols + col] - 0.0).abs() < 1e-6){
            Some(_) => true,
            None => false
        }
    }
    ///calculate reward for 4x4 
/// returns 1 if victory, -1 if loss, 0 otherwise
    pub fn calculate_reward(&self, row: usize, col:usize, player:i32)->f32{

        let directions: [(i32, i32); 4] = [
            (0, 1),   
            (1, 0),  
            (1, 1),  
            (1, -1),
        ];

        let mut max_count = -1;
        if let Some(_) = directions.iter().find(|&(x_dir,y_dir)|{
            let mut count = 1;
            let mut x= 1;
            while row as i32 - x * x_dir >= 0 && col as i32 - x * y_dir >= 0{

                if self.state[(row as i32 - x * x_dir) as usize * self.cols + (col as i32 - x * y_dir) as usize] == player as f32{
                    count += 1;
                    if count == 4{
                        return true
                    }
                }else{
                    break
                }
                x += 1;
            }
            x = 1;
            while ((row as i32 + x * x_dir) as usize) < self.rows && (col as i32 + x * y_dir) >= 0 && ((col as i32 + x * y_dir) as usize ) < self.cols {

                if self.state[(row as i32 + x * x_dir) as usize * self.cols + (col as i32 + x * y_dir) as usize] == player as f32{
                    count += 1;
                    if count == 4{
                        return true
                    }
                }else{
                    break
                }
                x += 1;
            }
            max_count = count.max(max_count);
            return false
        }){
            return player as f32
        }

        if max_count == 3{
            return 0.5
        }
        /* if self.calculate_reward(row, col, if player == 1{2} else {1}) == 0.5{
            return 0.5
        } */

        return 0.0
    }

    
}

impl Iterator for DeepQLearn{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = SmallRng::from_rng(&mut rand::rng());
        let rand = rng.random_range(0.0..1.0);
        let prev_state = self.state.clone();
        let action;
        let inserted;

        self.steps+=1;
        //with probability epsilon select random action. else choose the optimal action so far
        //execute action a
        if rand < self.epsilon {
            action = rng.random_range(0..self.cols);
            
            inserted = self.insert(action, if self.player == 1{Tile::Red} else{Tile::Blue});
            /* let mut tries = 1;
            while !inserted && tries <= 4{
                action = rng.random_range(0..self.cols);
                inserted = self.insert(action, if self.player == 1{Tile::Red} else{Tile::Blue});
                tries += 1;
            } */
            
        }
        else{
            //may have to check that state is initalized
            let actions: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> = self.action_value.forward(&self.state);

            //println!("{actions:?}");
            //let mut indices = Vec::new();
            let (_, indices) = actions.iter().enumerate().filter(|(idx, _)| self.is_action_valid(*idx)).fold((f32::NEG_INFINITY, Vec::new()), |(mx, mut indices), (idx, &val)|{if (mx - val).abs() < 1e-6 {indices.push(idx); (mx, indices)} else if val < mx {(mx, indices)} else {(val, vec![idx])}});
            if indices.len() == 0{
                return None
            }
            action = indices[rng.random_range(0..indices.len())];
            inserted = self.insert( action, if self.player == 1{Tile::Red} else{Tile::Blue});
            
        }
        //observe reward
        
        let reward = if inserted == -1{ 
            -0.95
        }else {
            let r = self.calculate_reward(inserted as usize, action, self.player);
            if r == (self.player as f32){
                1.0
            }else if self.calculate_reward(inserted as usize, action, if self.player == 1{2} else{1}) == 0.5{
                0.2
            }else if r == 0.0 || r == 0.5{
                0.0
            }else{
                -1.0
        }};

        //store transition
        self.replay_memory.push(ReplayTuple::new(prev_state.clone(), action, reward, self.state.clone(), (reward - 1.0).abs() < 1e-6 || (reward + 1.0).abs() < 1e-6));
        self.replay_memory.push(ReplayTuple::get_mirror(self.rows, self.cols, &prev_state, action, reward, &self.state, (reward - 1.0).abs() < 1e-6 || (reward + 1.0).abs() < 1e-6));

        
        if self.replay_memory.len() >= BATCH_SIZE && self.steps > WARMUP{
            //sample a random mini batch from replay memory
            //let batch = self.replay_memory.iter().choose_multiple(&mut rng, BATCH_SIZE);
            let batch: Vec<&ReplayTuple> = self.replay_memory.choose_multiple(&mut rng, BATCH_SIZE).collect();

            for ReplayTuple { state, action, reward, next_state, done } in batch {
                let mut next_q = self.target.forward(&next_state);
                for i in 0..self.cols{
                    if !self.is_action_valid(i){
                        next_q[i] = f32::NEG_INFINITY;
                    }
                }
                
                let max_next = 
                    if *done{
                        0.0
                    }else{
                        next_q.iter().fold(f32::NEG_INFINITY, |arg0: f32, other: &f32| f32::max(arg0, *other))
                    };

                //if next action terminates r, else r + gamma * max_next
                let target = reward + GAMMA as f32 * max_next;


                let (activations, /* pre_activations, */ mask) = self.action_value.forward_with_cache(&state);
                self.action_value.backward_and_update(&activations, /* &pre_activations, */&mask, *action, target, LEARNING_RATE);
            }
        }
        
        if ((reward - 1.0).abs() < 1e-6 || (reward + 1.0).abs() < 1e-6){
            println!("reward: {reward}");
            return None;
        }
        
        //TODO: represent actual reward
        return Some(reward)
    }
    
}