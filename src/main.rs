pub struct Board{
    board:Vec<Vec<Tile>>,
    size: usize
}

impl Board{
    fn new(size:usize)->Board{
        Board{board:vec![vec![Tile::Empty;size]; size], size}
    }
    fn insert(&mut self, col:usize, color: Tile)->bool{
        for row in (0..self.size).rev(){
            if self.board[row][col] == Tile::Empty{
                self.board[row][col] = color;
                return true
            }
        }
        false
    }
    fn clear(&mut self){
        (0..self.size).for_each(|row|{
            (0..self.size).for_each(|col|{
                self.board[row][col] = Tile::Empty;
            })
        });
    }
}

#[derive(PartialEq)]
#[derive(Clone)]
pub enum Tile{
    Empty,
    Red,
    Blue
}

fn main() {
    println!("Hello, world!");
    let mut board = Board::new(4);
    println!("{}", calculate_reward(&board));
    board.insert(0,Tile::Red);
    board.insert(0,Tile::Red);
    board.insert(0,Tile::Red);


    println!("{}", calculate_reward(&board));
    board.clear();
    board.insert(0, Tile::Blue);
    board.insert(1, Tile::Blue);
    board.insert(2, Tile::Blue);

    println!("{}", calculate_reward(&board));

    board.insert(1, Tile::Blue);
    board.insert(2, Tile::Blue);
    board.insert(2, Tile::Blue);
    board.insert(3, Tile::Red);
    board.insert(3, Tile::Red);
    board.insert(3, Tile::Red);
    board.insert(3, Tile::Blue);

    println!("{}", calculate_reward(&board));

}

///calculate reward for 4x4 
/// returns 1 if victory, -1 if loss, 0 otherwise
fn calculate_reward(state: &Board)->i32{
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
