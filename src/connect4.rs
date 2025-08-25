pub struct Board{
    pub board:Vec<Vec<Tile>>,
    pub size: usize
}

impl Board{
    pub fn new(size:usize)->Board{
        Board{board:vec![vec![Tile::Empty;size]; size], size}
    }
    pub fn insert(&mut self, col:usize, color: Tile)->bool{
        for row in (0..self.size).rev(){
            if self.board[row][col] == Tile::Empty{
                self.board[row][col] = color;
                return true
            }
        }
        false
    }
    pub fn clear(&mut self){
        (0..self.size).for_each(|row|{
            (0..self.size).for_each(|col|{
                self.board[row][col] = Tile::Empty;
            })
        });
    }
    pub fn flatten(&self)->Vec<i32>{
        self.board.iter().fold(Vec::new(), |mut acc, row|{
            row.iter().for_each(|cell|{
                acc.push(match &cell{
                    Tile::Empty => 0,
                    Tile::Red => 1,
                    Tile::Blue =>2
                });
            });
            acc
        })
    }
}

#[derive(PartialEq)]
#[derive(Clone)]

pub enum Tile{
    Empty,
    Red,
    Blue
}