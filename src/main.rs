mod connect4;
mod qlearn;
mod player2;

use std::collections::HashMap;

use connect4::{Board, Tile};
use player2::Player2;
use qlearn::{calculate_reward, QLearn};

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

    let mut player1 = QLearn::new(4, 1);
    //let mut player2 = QLearn::new(4,2);
    let mut player2 = Player2::new(HashMap::new(), Tile::Blue);
    let num_episodes = 4;

    for _ in 0..num_episodes{
        while let Some(_) = player1.next(){
            player2.turn(&mut player1);
        }
        player2 = Player2::new(player1.qtable.clone(), Tile::Blue);
    }

}


