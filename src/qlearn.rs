use std::collections::HashMap;

use crate::connect4::{Board, Tile};

pub fn intialize_qtable()->HashMap<Vec<i32>, i32>{
    HashMap::new()
}

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


struct qLearn{
    qtable: HashMap<Vec<i32>, i32>,
    epsilon: f64,
    state: Vec<i32>
}

/* impl Iterator for qLearn{
    type Item;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
} */