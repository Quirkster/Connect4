mod connect4;
mod qlearn;
mod player2;
mod gui;

use std::{collections::HashMap, io};

use connect4::{Board, Tile};
use gui::display;
use player2::Player2;
use qlearn::{calculate_reward, QLearn};
use stopwatch::Stopwatch;
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

    let mut player1 = QLearn::new(6, 7, 1);
    //let mut player2 = QLearn::new(4,2);
    let mut player2 = Player2::new(HashMap::new(), Tile::Blue);
    let num_episodes = 1000000;
    for episode in 0..num_episodes{
        //let name = format!("episode {episode}");
        //let rec = rerun::RecordingStreamBuilder::new(name).spawn().unwrap();
        //let mut turn_count = 0;
        let sw = Stopwatch::start_new();
        while let Some(_) = player1.next(){
            player2.turn(&mut player1);
            //display(&rec, &player1.state, player1.rows, player1.cols, turn_count);
            //turn_count += 1;
        }
        println!("episode {episode} completed in {:?}", sw.elapsed());
        if episode % 100 == 0{
            player2 = Player2::new(player1.qtable.clone(), Tile::Blue);
        }
        
    }
    let rec = rerun::RecordingStreamBuilder::new("final").spawn().unwrap();
    player2 = Player2::new(HashMap::new(), Tile::Blue);
    let mut turn_count = 0;
    while let Some(_) = player1.next(){
        //player2.turn(&mut player1);
        let mut input = String::new();

        println!("Enter an integer:");

        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let num: usize = input
            .trim()
            .parse()
            .expect("Input was not a valid integer");
        if num < player1.cols{
            player2.self_move(num as usize, &mut player1);
        }
        println!("You entered: {}", num);
        display(&rec, &player1.state, player1.rows, player1.cols, turn_count);
        turn_count += 1;
    }

}


