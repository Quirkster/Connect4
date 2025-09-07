mod connect4;
mod qlearn;
mod player2;
mod gui;
mod deepqlearn;
mod neuralnetwork;
mod player2deep;
mod fileops;

use std::{collections::HashMap, io};

use connect4::{Board, Tile};
use deepqlearn::{DeepQLearn, TARGET_UPDATE_FREQ};
use fileops::{load_layers, save_layers};
use gui::{display, display_deep};
use neuralnetwork::NeuralNetwork;
use player2::Player2;
use player2deep::Player2Deep;
use qlearn::{calculate_reward, QLearn, EPSILON_DECAY, EPSILON_MIN};
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

    
    let num_episodes = 100000;
    //deep_q_learn(num_episodes);
    deep_q_test(num_episodes);
    //q_learn(num_episodes);

}

fn q_learn(num_episodes:i32){
    let mut player1 = QLearn::new(6, 7, 1);
    //let mut player2 = QLearn::new(4,2);
    let mut player2 = Player2::new(HashMap::new(), Tile::Blue);
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

pub fn deep_q_test(num_tests:i32){
    let mut state = DeepQLearn::new(6,7,1);
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    let mut player1 = Player2Deep::new(NeuralNetwork::from_layers(load_layers("saved_weights7_4.bin").unwrap()),Tile::Blue);
    let mut player2 = Player2Deep::new(NeuralNetwork::from_layers(load_layers("saved_weights9_6.bin").unwrap()),Tile::Red);
    for episode in 0..num_tests{
        if episode % 2 == 0{
            let mut moves = 0;
            while moves < state.rows * state.cols{
                if player1.turn(&mut state ){
                    println!("player 1 win!");
                    player1_wins += 1;
                    break
                }if player2.turn(&mut state){
                    println!("player 2 win!");
                    player2_wins += 1;
                    break
                }
                moves += 2;
            }
            println!("{moves}");
            
            state.clear_board();
             
        }else{
            let mut moves = 0;
            while moves < state.rows * state.cols{
                if player2.turn(&mut state){
                    println!("player 2 win!");
                    player2_wins += 1;
                    break
                }
                if player1.turn(&mut state ){
                    println!("player 1 win!");
                    player1_wins += 1;
                    break
                }
                moves += 2;
            }
            
            state.clear_board();
        }

        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        //println!("{}", player1.epsilon);
        if episode % 4 == 0{
            
            player1.color = if player1.color == Tile::Red{Tile::Blue} else{Tile::Red};
            player2.color = if player2.color == Tile::Red{Tile::Blue} else{Tile::Red};
        }
        println!("episode {episode} completed");
    }

    println!("player 1: {player1_wins}, player 2:{player2_wins}");

}

pub fn deep_q_learn(num_episodes: i32){
    
    let sw_main = Stopwatch::start_new();
    let mut player1 = DeepQLearn::new(6, 7, 1);
    //player1.action_value = NeuralNetwork::from_layers(load_layers("saved_weights9_62.bin").unwrap());
    //player1.epsilon = 0.5;
    //let mut player2 = DeepQLearn::new(4,4,2);
    let mut player2 = Player2Deep::new(player1.action_value.clone_from(), Tile::Blue);
    //player2.qnet = NeuralNetwork::from_layers(load_layers("saved_weights9_62.bin").unwrap());
    for episode in 0..num_episodes{
        //let name = format!("episode {episode}");
        //let rec = rerun::RecordingStreamBuilder::new(name).spawn().unwrap();
        //let mut turn_count = 0;
        let sw = Stopwatch::start_new();
        //let mut turn_count = 0;
        if episode % 2 == 0{
             while let Some(_) = player1.next(){
            
                if player2.turn(&mut player1){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
        }else{
            player2.turn(&mut player1);
            while let Some(_) = player1.next(){
            
                if player2.turn(&mut player1){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
        }

        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        println!("episode {episode} completed in {:?}", sw.elapsed());
        if episode % TARGET_UPDATE_FREQ == 0 {
           player1.target = player1.action_value.clone_from();
        }
        if episode % 50 == 0{
           player2 = Player2Deep::new(player1.action_value.clone_from(), player2.color) ;
        }
        // Decay epsilon
        player1.epsilon = (player1.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        //println!("{}", player1.epsilon);

        player1.clear_board();
        if episode % 4 == 0{
            player1.player = if player1.player == 1{2} else{1};
            player2.color = if player2.color == Tile::Red{Tile::Blue} else{Tile::Red};
        }

        
    }
    println!("{num_episodes} episodes completed in {:?}", sw_main.elapsed());
    let _ = save_layers("saved_weights.bin", &player1.action_value.layers);
    //player2 = Player2::new(HashMap::new(), Tile::Blue);
    let rec = rerun::RecordingStreamBuilder::new("final").spawn().unwrap();

    let mut turn_count = 0;
    while let Some(_) = player1.next(){

        display_deep(&rec, &player1.state, player1.rows, player1.cols, 2*turn_count);
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
            if player2.self_move(num as usize, &mut player1){
                println!("You win!");

                display_deep(&rec, &player1.state, player1.rows, player1.cols, 2*turn_count + 1);
                break
            }
        }
        println!("You entered: {}", num);
        display_deep(&rec, &player1.state, player1.rows, player1.cols, 2*turn_count + 1);
        turn_count += 1;
    }
    
}


