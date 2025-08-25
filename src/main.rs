mod connect4;
mod qlearn;

use connect4::{Board, Tile};
use qlearn::calculate_reward;

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


