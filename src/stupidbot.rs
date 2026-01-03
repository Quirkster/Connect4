use rand::{rng, Rng};
use rand_distr::{Distribution, StandardNormal};
use rand_distr::weighted::WeightedIndex;

use crate::connect4::Tile;
use crate::deepqlearn::{DeepQLearn, ReplayTuple};
use crate::neuralnetwork::NeuralNetwork;
use crate::qlearn::QLearn;
pub struct StupidBot{
    pub qnet: NeuralNetwork,
    pub color: Tile,
    pub col: usize,
    pub horiz: bool,
    pub turns:usize,
}

impl StupidBot{
    pub fn new(qnet: NeuralNetwork, color:Tile)->StupidBot{
        let mut rng = rand::rng();
        StupidBot{qnet, color, col:rng.random_range(0..7), horiz: rng.random_bool(0.5), turns:0}
    }
    pub fn turn(&mut self, state: &mut DeepQLearn)->bool{
        self.turns += 2;
        if self.turns > 42{
            return false
        }
        /* if let Some(actions) = self.qnet.forward(&state.state){ */
        let player_num = match self.color{
            Tile::Red => 1.0,
            Tile::Blue => -1.0,
            Tile::Empty => 0.0,
        };
        let mut rng = rand::rng();
        if !self.horiz{
            while !state.is_action_valid(self.col){
                self.col = rng.random_range(0..7);
            }
        }else{
            while !state.is_action_valid(self.col){
                
                self.col += 1;
                if self.col == 7{
                    self.col = 0;
                }
            }
        }
        
        //println!("{:?}", actions);
        
        
        let prev_state = state.state.clone();
        let inserted = state.insert(self.col, self.color.clone());
        if (state.calculate_reward(inserted as usize, self.col, player_num as i32) - player_num).abs() < 1e-6{
            state.replay_memory.push(ReplayTuple::new(prev_state, 0, -1.0, state.state.clone(), true));
            if self.horiz{
                self.col += 1;
                if self.col == 7{
                    self.col = 0;
                }
            }
            return true
        }
        /* }else{
            let mut rng = rng();
            state.insert(rng.random_range(0..state.cols), self.color.clone());
        } */
       self.col += 1;
       if self.col == 7{self.col = 0;}
       false
    
    }
    pub fn switch_column(&mut self){
        let mut rng = rand::rng();
        self.col = rng.random_range(0..7);
        self.horiz = rng.random_bool(0.5);
        self.turns = 0;

    }
}