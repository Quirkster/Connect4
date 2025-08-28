use std::collections::HashMap;
use rand::prelude::*;
use crate::connect4::{Board, Tile};

const EPSILON_MIN:f64 = 0.1;
const EPSILON_DECAY:f64 = 0.9;
const ALPHA: f64 = 0.3;
const GAMMA: f64 = 0.9;
///calculate reward for 4x4 
/// returns 1 if victory, -1 if loss, 0 otherwise
pub fn calculate_reward(state: &Board)->i32{
    if state.board.contains(&vec![Tile::Red;state.size]){
        return 1
    }
    if state.board.contains(&vec![Tile::Blue; state.size]){
        return -1
    }
    for col in 0..state.size{
        match state.board.iter().fold((true, true), |(mut won, mut lost), row|{
            if row[col] != Tile::Red{
                won = false;
            }if row[col] != Tile::Blue{
                lost = false;
            }
            return (won, lost)
        }){
            (false, false) => (),
            (false, true) => return -1,
            (true, false) => return 1,
            _ => panic!("won and lost")
        }
    }
    let (mut won, mut lost) = (true, true);
    let (mut won_neg, mut lost_neg) = (true, true);
    for index in 0..state.size{
        if state.board[index][index] != Tile::Red{
            won = false;
        }if state.board[index][index] != Tile::Blue{
            lost = false;
        }
        if state.board[index][state.size-index - 1] != Tile::Red{
            won_neg = false;
        }if state.board[index][state.size-index- 1] != Tile::Blue{
            lost_neg = false;
        }
    }
    if won || won_neg{
        return 1
    }if lost || lost_neg{
        return -1
    }


    return 0
}


pub struct QLearn{
    pub qtable: HashMap<Vec<i32>, Vec<f64>>,
    pub epsilon: f64,
    pub state: Vec<i32>,
    pub rows: usize,
    pub cols:usize,
    pub player: i32,
}

impl QLearn{
    pub fn new(rows:usize, cols:usize, player: i32)->QLearn{
        QLearn{qtable:HashMap::new(), epsilon:1.0, state:vec![0;rows * cols], rows, cols, player}
    }
    pub fn insert(&mut self, col:usize, color: Tile)->bool{
        for row in (0..self.rows).rev(){
            if self.state[row * self.cols + col] == 0{
                self.state[row * self.cols + col] = match color {
                    Tile::Red => 1,
                    Tile::Blue => 2,
                    Tile::Empty => 0,
                };
                println!("inserted into col: {col}");
                return true
            }
        }
        false
    }
    pub fn clear_board(&mut self){
        (0..(self.rows*self.cols)).for_each(|i|{
            self.state[i] = 0;
        })
    }
    pub fn unflatten(&self)->Vec<Vec<Tile>>{
        let mut res = vec![Vec::with_capacity(self.cols); self.rows];
        for row in 0..self.rows{
            for col in 0..self.cols{
                match self.state[row * self.cols + col]{
                    0 => res[row].push(Tile::Empty),
                    1 => res[row].push(Tile::Red),
                    2 => res[row].push(Tile::Blue),
                    _ => panic!("invalid tile")
                }
            }
        }
        res
    }
    ///calculate reward for 4x4 
/// returns 1 if victory, -1 if loss, 0 otherwise
    pub fn calculate_reward(&self)->i32{
        let (rows, cols) = (self.rows, self.cols);

        let directions: [(i32, i32); 4] = [
            (0, 1),   
            (1, 0),  
            (1, 1),  
            (1, -1),
        ];

        for r in 0..rows{
            for c in 0..cols{
                if self.state[r * cols + c] == 0{
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

        return 0
    }
}

impl Iterator for QLearn{
    type Item = Vec<i32>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = rand::rng();
        let rand = rng.random_range(0.0..1.0);
        let actions_exists = self.qtable.contains_key(&self.state);
        let prev_state = self.state.clone();
        if !actions_exists{
            self.qtable.insert(self.state.clone(), vec![0.0; self.cols]);
            //ctions = self.qtable.get(&self.state);
        }
        let action;
        if rand < self.epsilon || actions_exists{
            action = rng.random_range(0..self.cols);
            
            self.insert(action, if self.player == 1{Tile::Red} else{Tile::Blue});
            
        }
        else{
            //may have to check that state is initalized
            let (max_index, _) = self.qtable.get(&self.state).expect("state not in table").iter().enumerate().fold((self.rows*self.cols , std::f64::NEG_INFINITY), |(max_index, max), (index, &val)|{
                if val > max{
                    (index, val)
                }else{
                    (max_index, max)
                }
            });
            action = max_index;
            self.insert( max_index, if self.player == 1{Tile::Red} else{Tile::Blue});
            
        }
        let reward = if self.calculate_reward() == self.player{
            1
        }else if self.calculate_reward() == 0{
            0
        }else{
            -1
        };
        println!("{reward}");
        let (_, next_reward) = if let Some(acts) = self.qtable.get(&self.state){
            acts.iter().enumerate().fold((self.rows*self.cols, std::f64::NEG_INFINITY), |(max_index, max), (index, &val)|{
                if val > max{
                    (index, val)
                }else{
                    (max_index, max)
                }
            })
        }else{
            (0, 0.0)
        };
        let prev_reward = self.qtable.get_mut(&prev_state).expect("invalid state");

        prev_reward[action] += ALPHA * (reward as f64 + GAMMA * next_reward - prev_reward[action]);
        if reward >= 1 || reward <= -1 || !self.state.contains(&0){
            //self.epsilon = 1.0;
            self.clear_board();
            return None
        }
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        return Some(self.state.clone())
    }
}