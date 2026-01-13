mod connect4;
mod qlearn;
mod player2;
mod gui;
mod deepqlearn;
mod neuralnetwork;
mod player2deep;
mod fileops;
mod nngui;
mod stupidbot;
mod alphabeta;
mod unbox;

use std::{collections::HashMap, io};

use connect4::{Board, Tile};
use deepqlearn::{DeepQLearn, TARGET_UPDATE_FREQ, EXPLORATION};
use fileops::{load_layers, save_layers};
use gui::{display, display_deep};
use ndarray::Array1;
use neuralnetwork::NeuralNetwork;
use player2::Player2;
use player2deep::Player2Deep;
use qlearn::{calculate_reward, QLearn, EPSILON_DECAY, EPSILON_MIN};
use rand::Rng;
use stopwatch::Stopwatch;

use crate::{alphabeta::MCTSNode, stupidbot::StupidBot};
fn main() {

    let num_episodes = 10000;
    //deep_q_learn(num_episodes);
    //deep_q_test(num_episodes/10);
    //q_learn(num_episodes);

    play(&String::from("saved_weights.bin"));

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
}

pub fn deep_q_test(num_tests:i32){
    let mut state = DeepQLearn::new(6,7,1);
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    let mut player1 = Player2Deep::new(NeuralNetwork::from_layers(load_layers("saved_weights.bin").unwrap()),Tile::Blue);
    let mut player2 = Player2Deep::new(NeuralNetwork::from_layers(load_layers("saved_weights1_3_3.bin").unwrap()),Tile::Red);
    //let mut player2 = Player2Deep::new(NeuralNetwork::from_layers(load_layers("saved_weights12_27.bin").unwrap()),Tile::Red);
    //let mut player2 = StupidBot::new(NeuralNetwork::from_layers(load_layers("saved_weights1_3_2.bin").unwrap()), Tile::Blue);
    //let mut player2 = Player2Deep::new(NeuralNetwork::new(42, &[64,64], 7),Tile::Red);

    for episode in 0..num_tests{
        if episode % 2 == 0{
            let mut moves = 0;
            while moves < state.rows * state.cols{
                if player1.turn(&mut state, false ){
                    println!("player 1 win!");
                    player1_wins += 1;
                    break
                }if player2.turn(&mut state, true){
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
                if player2.turn(&mut state, true){
                    println!("player 2 win!");
                    player2_wins += 1;
                    break
                }
                if player1.turn(&mut state , false ){
                    println!("player 1 win!");
                    player1_wins += 1;
                    break
                }
                moves += 2;
            }
            
            //player2.turns = 0;
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


    /* if player2_wins > player1_wins{
        let _ = save_layers("randomly_good_weights.bin", &player2.qnet.layers);

    } */

}

pub fn deep_q_learn(num_episodes: i32){
    
    let sw_main = Stopwatch::start_new();
    let mut training_agent = DeepQLearn::new(6, 7, 1);


    //random_weights
    let mut random_weights_agent = Player2Deep::new(training_agent.action_value.clone_from(), Tile::Blue);
    random_weights_agent.randomize_weights();


    //training bot
    training_agent.action_value = NeuralNetwork::from_layers(load_layers("saved_weights1_3_3.bin").unwrap());
    training_agent.epsilon = 1.0;
    
    //previous version of itself
    let mut frozen_agent = Player2Deep::new(training_agent.action_value.clone_from(), Tile::Blue);
    frozen_agent.qnet = NeuralNetwork::from_layers(load_layers("saved_weights1_3_3.bin").unwrap());


    let mut current_best_bot = Player2Deep::new(training_agent.action_value.clone_from(), Tile::Blue);
    current_best_bot.qnet = NeuralNetwork::from_layers(load_layers("saved_weights1_3_3.bin").unwrap());

    //aggressively bad player(aims to immediately get connect 4 by filling up an entire column or row)
    let mut stupidbot = StupidBot::new(training_agent.action_value.clone_from(), Tile::Blue);
    for episode in 0..num_episodes{
        //let name = format!("episode {episode}");
        //let rec = rerun::RecordingStreamBuilder::new(name).spawn().unwrap();
        //let mut turn_count = 0;

        let mut rng = rand::rng();
        let num:f32 = rng.random();
        let opponent = if num < 0.4{
            &frozen_agent
        } else if num > 0.8{
            &random_weights_agent
        }else{
            &current_best_bot
        };
        let sw = Stopwatch::start_new();
        //let mut turn_count = 0;
        if episode % 2 == 0{
             while let Some(_) = training_agent.next(){
            
                if opponent.turn(&mut training_agent, true ){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
        }else{
            opponent.turn(&mut training_agent, true);
            while let Some(_) = training_agent.next(){
            
                if opponent.turn(&mut training_agent, true){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
        }

        //have it play a stupid bot 10% that just inserts in the same column/row so that it learns what connect 4 looks like
        //since the turn function needs to be so different it has to be a different time and cant be randomly rotated like the others
        if episode % 10 == 0{
            training_agent.clear_board();
             while let Some(_) = training_agent.next(){
            
                if stupidbot.turn(&mut training_agent ){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
            stupidbot.switch_column();
        }else if episode % 5 == 0{
            training_agent.clear_board();
            stupidbot.turn(&mut training_agent);
            while let Some(_) = training_agent.next(){
            
                if stupidbot.turn(&mut training_agent){
                    break
                }
                //turn_count += 1;
                //display_deep(&rec, &player1.state, player1.rows, player1.cols, turn_count);
                //turn_count += 1;
            }
            stupidbot.switch_column();
        }

        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        println!("episode {episode} completed in {:?}", sw.elapsed());
        if episode % TARGET_UPDATE_FREQ == 0 {
           training_agent.target = training_agent.action_value.clone_from();
        }
        if episode % 500 == 0{
           frozen_agent = Player2Deep::new(training_agent.action_value.clone_from(), frozen_agent.color) ;
           random_weights_agent.randomize_weights();
        }
        // Decay epsilon
        training_agent.epsilon = (training_agent.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        //println!("{:?}, {:?}", player1.action_value, player2.qnet);
        //println!("{}", player1.epsilon);
        if training_agent.steps > EXPLORATION as usize {
            training_agent.epsilon = (training_agent.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        }

        stupidbot.turns = 0;
        training_agent.clear_board();/* 
        if episode % 4 == 0{
            player1.player = if player1.player == 1{2} else{1};
            player2.color = if player2.color == Tile::Red{Tile::Blue} else{Tile::Red};
        } */

        
    }
    println!("{num_episodes} episodes completed in {:?}", sw_main.elapsed());
    let _ = save_layers("saved_weights.bin", &training_agent.action_value.layers);
    //player2 = Player2::new(HashMap::new(), Tile::Blue);
    /* let rec = rerun::RecordingStreamBuilder::new("final").spawn().unwrap();

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
    } */
}

pub fn play(weights: &String){
    let rec = rerun::RecordingStreamBuilder::new("final").spawn().unwrap();
    let player1 = Player2Deep::new(NeuralNetwork::from_layers(load_layers(weights).unwrap()), Tile::Blue);
    //player1.qnet = NeuralNetwork::from_layers(load_layers("saved_weights12_25_3.bin").unwrap());

    let player2 = Player2Deep::new(NeuralNetwork::from_layers(load_layers(weights).unwrap()),Tile::Red);
    let mut state = DeepQLearn::new(6,7,1);

    let mut moves = 0;
    while moves < state.rows * state.cols{
        println!("P1 Suggested moves: {:?}", player1.qnet.forward(&state.state));

        /*if player1.turn(&mut state, false){
            display_deep(&rec, &state.state, state.rows, state.cols, moves as i32);
            println!("You lose!");
            break
        } */
        let action = MCTSNode::mcts_search(state.state.to_owned(), 10000);

        if player1.self_move(action as usize, &mut state){
            display_deep(&rec, &state.state, state.rows, state.cols, moves as i32);
            println!("You lose!");
            break
        }
       
       display_deep(&rec, &state.state, state.rows, state.cols, moves as i32);


        println!("Suggested moves: {:?}", player2.qnet.forward(&state.state));
        let mut input = String::new();

        println!("Enter an integer:");

        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let num: usize = input
            .trim()
            .parse()
            .expect("Input was not a valid integer");
        if num < state.cols{
            if player2.self_move(num as usize, &mut state){
                println!("You win!");

                display_deep(&rec, &state.state, state.rows, state.cols, moves as i32 + 1);
                break
            }
        }
        println!("You entered: {}", num);
        display_deep(&rec, &state.state, state.rows, state.cols, moves as i32+ 1);
        moves += 2;
    }

        //display_deep(&rec, &player1.state, player1.rows, player1.cols, 2*turn_count);
        //player2.turn(&mut player1);

}


//[0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]