use crate::connect4::Tile;
use crate::deepqlearn::DeepQLearn;
use crate::neuralnetwork::NeuralNetwork;
use crate::qlearn::QLearn;
pub struct Player2Deep{
    pub qnet: NeuralNetwork,
    pub color: Tile
}

impl Player2Deep{
    pub fn new(qnet: NeuralNetwork, color:Tile)->Player2Deep{
        Player2Deep{qnet, color}
    }
    pub fn turn(&self, state: &mut DeepQLearn){
        /* if let Some(actions) = self.qnet.forward(&state.state){ */
        let actions = self.qnet.forward(&state.state);
        let (max_index, _) = actions.iter().enumerate().fold((state.rows*state.cols, std::f32::NEG_INFINITY), |(max_index, max), (index, &val)|{
            if val > max{
                (index, val)
            }else{
                (max_index, max)
            }
        });
        state.insert(max_index, self.color.clone());

        /* }else{
            let mut rng = rng();
            state.insert(rng.random_range(0..state.cols), self.color.clone());
        } */
    }

    pub fn self_move(&self, col:usize, state: &mut DeepQLearn){
        state.insert(col, self.color.clone());
    }
}