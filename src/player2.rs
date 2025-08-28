use std::collections::HashMap;

use rand::{rng, Rng};

use crate::connect4::Tile;
use crate::qlearn::QLearn;
pub struct Player2{
    qtable: HashMap<Vec<i32>, Vec<f64>>,
    color: Tile
}

impl Player2{
    pub fn new(qtable:HashMap<Vec<i32>, Vec<f64>>, color:Tile)->Player2{
        Player2{qtable, color}
    }
    pub fn turn(&self, state: &mut QLearn){
        if let Some(actions) = self.qtable.get(&state.state){
            let (max_index, _) = actions.iter().enumerate().fold((state.rows*state.cols, std::f64::NEG_INFINITY), |(max_index, max), (index, &val)|{
                if val > max{
                    (index, val)
                }else{
                    (max_index, max)
                }
            });
            state.insert(max_index, self.color.clone());

        }else{
            let mut rng = rng();
            state.insert(rng.random_range(0..state.cols), self.color.clone());
        }
    }

    pub fn self_move(&self, col:usize, state: &mut QLearn){
        state.insert(col, self.color.clone());
    }
}