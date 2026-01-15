use std::{mem, ops::DerefMut};

use ndarray::Array1;
use rand::Rng;

use crate::{connect4::Tile, qlearn::calculate_reward};

use std::cell::RefCell;
use std::rc::{Rc, Weak};


//Source binary_tree rust crate to learn about how trees work
/* pub trait MyTree {
    type MyNode: MyNode;

    fn root(&self) -> Option<&Self::MyNode>;
} */

unsafe fn borrow<'a, T, U>(raw: *const T, _: &'a U) -> &'a T {
    &*raw
}

unsafe fn borrow_mut<'a, T, U>(raw: *mut T, _: &'a U) -> &'a mut T {
    &mut *raw
}

/* pub trait MyNode{
    type Value;

    fn children(&self)->Vec<Option<&Self>>;

    fn child(&self, action: usize)->Option<&Self>;

    fn state(&self)->&Array1<f32>;

    fn parent(&self) ->Option<&Self>;

    fn player(&self)->i32;
 
    fn visits(&self)->usize;

    fn wins(&self)->usize;

    fn untried_actions(&self)->&Vec<usize>;

    fn walk<'a, F>(&'a self, mut step_in: F)
        where F: FnMut(&'a Self) -> usize{

            let mut subtree = Some(self);
            while let Some(mut st) = subtree{
                let action = step_in(&mut st);
                subtree = st.child(action);
                //if game over break
            }
        }

} */

/* pub trait NodeMut: MyNode + Sized {
    type NodePtr: Sized + DerefMut<Target = Self>;

    fn detach(&mut self, action:usize) -> Option<Self::NodePtr>;

    fn new_level(&mut self, level:Vec<Option<Self::NodePtr>>);//->Self::NodePtr

    fn value_mut(&mut self) -> &mut Self::Value;

    fn into_parts(self) ->(Self::Value, Vec<Option<Self::NodePtr>>);

    fn child_mut(&mut self, action:usize)->Option<&mut Self>;

    /// Simple mutable walk
    ///
    /// Note that the type of `step_in` is almost identical to that in
    /// `Node::walk`, but not exactly so. Here, `step_in` does not get a
    /// reference which lives as long as `self` so that it cannot leak
    /// references out to its environment.
    /// 
    fn walk_mut<'a, FI, FS>(&'a mut self, mut step_in: FI, stop: FS)
        where FI: FnMut(&Self) -> i32,
              FS: FnOnce(&'a mut Self)
    {
        let mut node: *mut _ = self;
        loop{
            let action = {
                let pin = ();
                step_in(unsafe{borrow(node, &pin)})
            };

            if action < 0{
                break
            }
            let next = unsafe { borrow_mut(node, self) }.child_mut(action as usize);
            
            if let Some(st) = next{
                node = st;
            }else{
                break;
            }

        }
        stop(unsafe { borrow_mut(node, self) });

    }
} */

//pub type NodePtr<T> = Box<MCTSNode<T>>;
pub type NodePtr<MCTSNode> = Rc<RefCell<MCTSNode>>;


#[derive(Debug)]
pub struct MonteCarloSearchTree(Option<NodePtr<MCTSNode>>);

impl MonteCarloSearchTree{
    /* fn root_must(&mut self)-> &mut RefCell<MCTSNode<T>>{
        &mut **self.0.as_mut().unwrap()
    } */
    pub fn new() -> MonteCarloSearchTree{
        MonteCarloSearchTree(None)
    }
    pub fn is_empty(&self) -> bool{
        self.0.is_none()
    }



}

/* impl<T> MyTree for MonteCarloSearchTree<T>{
    type MyNode = MCTSNode<T>;
    fn root(&self) -> Option<&Self::MyNode>{
        self.0.as_ref().map(|nodeptr| &**nodeptr)
    }



} */

#[derive(Debug)]
pub struct MCTSNode {
    parent: Option<Weak<RefCell<MCTSNode>>>,
    children: Vec<Option<NodePtr<MCTSNode>>>,
    player: i32,
    visits: usize,
    wins: f32,
    untried_actions: Vec<usize>,
    state: Array1<f32>,
    rows: usize,
    cols: usize,
    level: usize,
    action: Option<usize>

}
impl MCTSNode{
    fn new(
        parent: Option<Weak<RefCell<MCTSNode>>>,
        children: Vec<Option<NodePtr<MCTSNode>>>,
        player: i32,
        visits: usize,
        wins: f32,
        rows: usize,
        cols: usize,
        level: usize,
        action: Option<usize>,
        state: Array1<f32>)->MCTSNode{
        MCTSNode{parent, children, player, level, visits, wins, untried_actions:Self::available_actions(&state, rows, cols), state,rows, cols, action }
    }

    /* pub fn selection(&self){
        //s0.. s6 = possible moves(inserting into columns 0..7)
        self.untried_actions.iter().fold(Vec::new(), |mut acc, action|{
            let mut new_state = self.state.clone();
            insert(new_state, action);
            acc.push(new_state);
            acc
        })


        //for root both will be unvisited so UCB = infinity and pick one randomly
    }

    pub fn expansion(){


    }

    pub fn simulation(&self, ){
        //simulate starting at chosen state
    }

    pub fn backpropagation(&self, actions:Vec<usize>){
        //go backwards through actions and "undo them" while keeping track of the result of each one

    }
    pub fn ucb1(&self)->f32{
        //i =self.level
        //Xi + self.c + sqrt(ln(N)/ni)
        0.0
    } */
    pub fn can_insert(state: &Array1<f32>, cols: usize, col:usize)->bool{

        if (state[col] - 0.0).abs() < 1e-6{
            true
        }
        else{
            false
        }
    }
    pub fn insert(state: &mut Array1<f32>, rows: usize, cols: usize, col:usize, color: f32)->i32{
        for row in (0..rows).rev(){
            if (state[row * cols + col] - 0.0).abs() < 1e-6{
                state[row * cols + col] = color;
                return row as i32
            }
        }
        -1
    }

    pub fn calculate_result(&self)->Result{
        //vertical connect 4;
        for c in 0..self.cols{
            let mut sum:f32 = (0..4).map(|i| { self.state[i * self.cols + c] }).sum();
            let mut r = 4;

            if (sum - 4.0).abs() < 1e-5{
                return Result::WIN
            }else if (sum + 4.0).abs() < 1e-5{
                return Result::LOSE
            }
            while r < self.rows{
                sum -= self.state[(r - 4) * self.cols + c];

                sum += self.state[r * self.cols + c];

                if (sum - 4.0).abs() < 1e-5{
                    return Result::WIN
                }else if (sum + 4.0).abs() < 1e-5{
                    return Result::LOSE
                }

                r += 1
                
            }

        }
        //check horizontal
        for r in 0..self.rows{
            let mut sum:f32 = (0..4).map(|i| { self.state[r * self.cols + i] }).sum();
            let mut c = 4;

            if (sum - 4.0).abs() < 1e-5{
                return Result::WIN
            }else if (sum + 4.0).abs() < 1e-5{
                return Result::LOSE
            }
            while c < self.cols{
                sum -= self.state[r * self.cols + c - 4];

                sum += self.state[r * self.cols + c];

                if (sum - 4.0).abs() < 1e-5{
                    return Result::WIN
                }else if (sum + 4.0).abs() < 1e-5{
                    return Result::LOSE
                }

                c += 1
                
            }

        }
        for start_row in 0..(self.rows - 3){
            for start_col in 0..(self.cols - 3){
                let sum:f32 = (0..4).map(|i| { self.state[(start_row + i) * self.cols + start_col + i] }).sum();
                if (sum - 4.0).abs() < 1e-5{
                    return Result::WIN
                }else if (sum + 4.0).abs() < 1e-5{
                    return Result::LOSE
                }

            }
        }

        for start_row in (3..self.rows).rev(){
            for start_col in 0..(self.cols - 3){
                let sum:f32 = (0..4).map(|i| { self.state[(start_row - i) * self.cols + start_col + i] }).sum();
                if (sum - 4.0).abs() < 1e-5{
                    return Result::WIN
                }else if (sum + 4.0).abs() < 1e-5{
                    return Result::LOSE
                }

            }
        }

        if self.is_draw(){
            return Result::DRAW
        }

        //println!("not a draw");
        return Result::INPROGRESS
    }

    pub fn is_draw(&self)->bool{
        //println!("{:?}", self.untried_actions);
        //println!("{}, {}", self.untried_actions.len(), self.state.sum());
        !self.state.iter().any(|&row|{
            if row != 0.0{
                return false
            }
            true
        })
    }

    pub fn available_actions(state: &Array1<f32>, rows:usize, cols:usize)->Vec<usize>{
        //println!("available actions");
        (0..cols).fold(Vec::<usize>::new(), |mut acc:Vec<usize>, col|{
            if Self::can_insert(state, cols, col){
                acc.push(col);
            }
            return acc
        })
    }

    pub fn is_terminal(&self)->bool{
        self.calculate_result() != Result::INPROGRESS
    }

    pub fn is_fully_expanded(&self)->bool{
        //println!("is_fully_expanded");
        self.untried_actions.len() == 0
    }

    pub fn expand(root:&NodePtr<MCTSNode>)->NodePtr<MCTSNode>{
        //println!("expand");
        let action;
        {
            let mut r = root.borrow_mut();
            action = r.untried_actions.pop().unwrap();

        }
        let child;
        {
        let r = root.borrow();
        let mut new_state = r.state.to_owned();
        Self::insert(&mut new_state, r.rows, r.cols, action, -r.player as f32);
        child = Rc::new(RefCell::new(MCTSNode::new( Some(Rc::downgrade(root)), Vec::new(), -r.player, 0, 0.0, r.rows, r.cols, r.level + 1, Some(action), new_state)));
        }
        {
            let mut r = root.borrow_mut();
         
            r.children.push(Some(Rc::clone(&child)));
        }
        return child
        
    }

    pub fn best_child(&self)->Option<NodePtr<MCTSNode>>{
        //println!("best_child");
        for c in &self.children{
            if let Some(child) = c{
                if child.borrow().visits == 0{
                    return Some(Rc::clone(child))
                }
            }
            
        }
        let max_children = self.children.iter()
            .filter_map(|c| c.as_ref())
            .fold((Vec::new(), -1.0), |(mut maxes, max_ucb),a|{
                let a_ref = a.borrow();
                let a_ucb = self.ucb(a_ref.wins, a_ref.visits);
                if a_ucb > max_ucb{
                    (vec![Rc::clone(a)], a_ucb)
                }else if (a_ucb - max_ucb).abs() < 1e-6{
                    maxes.push(Rc::clone(a));
                    (maxes, max_ucb)
                }else{
                    (maxes, max_ucb)
                }
                //a_ucb.partial_cmp(&b_ucb).unwrap_or(std::cmp::Ordering::Equal)
            });
        if max_children.0.len() == 0{
            return None
        }
        let mut rng = rand::rng();
        let best_child = rng.random_range(0..max_children.0.len());
        return Some(Rc::clone(&max_children.0[best_child]));
    }
    
    fn ucb(&self, child_wins:f32, child_visits: usize)-> f32{
        //println!("ucb");
        let c = 1.4;
        let exploit = child_wins as f32/ child_visits as f32;
        let explore = c * ((self.visits as f32).ln()/ (child_visits as f32)).powf(0.5);
        return exploit + explore
    }

    pub fn rollout(&self)->i32{
        //println!("rollout");
        let mut new_state = self.state.to_owned();

        let mut player = self.player;

        loop{
            let winner = self.calculate_result();
            if winner != Result::INPROGRESS{
                if winner == Result::DRAW{
                    //println!("DRAW(rollout)");
                    return 0
                }else {
                    //println!("winner: {player}(rollout)");
                    return player
                }
            }

            //let winner = self.calculate_result(col, row, player)
            let actions = Self::available_actions(&new_state, self.rows, self.cols);
            if actions.len() == 0{
                return 0
            }
            let mut rng = rand::rng();
            let move_idx = rng.random_range(0..actions.len());

            Self::insert(&mut new_state, self.rows, self.cols, actions[move_idx], player as f32);


            player = -player;


        }
    }

    pub fn backpropagate(&mut self, winner:i32){
        //println!("backpropagate");
        self.visits += 1;
        if winner == 0{
            self.wins += 0.5
        }else if winner == self.player{
            self.wins += 1.0
        }

        if let Some(p) = &self.parent {
            if let Some(parent_rc) = p.upgrade() {
                parent_rc.borrow_mut().backpropagate(winner);
            }
        }
    }
    pub fn mcts_search(root_state:Array1<f32>, iterations:usize)->usize{
        let root: Rc<RefCell<MCTSNode>>= Rc::new(RefCell::new(MCTSNode::new(None,Vec::new(), 1, 0, 0.0, 6, 7, 0,None, root_state )));
        if root.borrow().is_terminal(){
            panic!("OOPSY");
        }
        for _ in 0..iterations{
            //println!("{}", root.borrow().is_terminal());
            let mut node: Rc<RefCell<MCTSNode>> = Rc::clone(&root);
            while !&node.borrow().is_terminal() && node.borrow().is_fully_expanded(){
                let next = {

                    let n = node.borrow();
                    n.best_child().unwrap()
                };
                node = next;


            }
            //println!("{:?}", node.borrow().calculate_result());
            if !node.borrow().is_terminal() && !node.borrow().is_fully_expanded(){
                node = MCTSNode::expand(&node);
            }
            let winner = node.borrow().rollout();
            node.borrow_mut().backpropagate(winner);
        }

        {
        let r = Rc::clone(&root);
        let r_borrow = r.borrow();

        let act;
        {
            //println!("{}, {}", iterations, r_borrow.children.len());
            //println!("r_borrow {r_borrow:?}");
            let best_lst = r_borrow.children.iter().rev().fold((Vec::new(), -1), |(mut acc, max_visits), child|
            {
                let c_ref = child.as_ref();
                let visits = c_ref.unwrap().borrow().visits as i32;
                if visits == max_visits{
                    acc.push(Rc::clone(&c_ref.unwrap()));
                    (acc, max_visits)
                }else if visits > max_visits{
                    (vec![Rc::clone(&c_ref.unwrap())], visits)
                }else{
                    (acc, max_visits)
                }

            });
            let best;
            if best_lst.0.len() == 0{
                best = None
            }else{
                let mut rng = rand::rng();
                let best_idx = rng.random_range(0..best_lst.0.len());
                best = Some(Rc::clone(&best_lst.0[best_idx]));
            }

            /* r_borrow.children.iter().for_each(|child|{
                let a = child.as_ref();
                println!("{:?}, {}, {}", a.unwrap().borrow().action, a.unwrap().borrow().visits, a.unwrap().borrow().wins);
            }); */

            let best_ref  = best.as_ref();

            //println!("{}, {:?}", best_lst.0.len(), best_ref.unwrap().borrow().action);


            act = match best_ref{
                Some(b) => {
                    let x = b.as_ref();
                    x.borrow().action.unwrap()

                 }
                 None => {let mut rng = rand::rng();
                    let idx = rng.random_range(0..r_borrow.untried_actions.len());
                    r_borrow.untried_actions[idx]
                }
            };

        }
        act
    }
        
    }

    

    




}

impl MCTSNode{
    fn children(&self)->Vec<Option<&RefCell<Self>>>{
        self.children.iter().fold(Vec::new(), |mut acc, child|{acc.push(child.as_ref().map(|st| &**st)); acc})
    }

    fn child(&self, action: usize)->Option<&RefCell<Self>>{
        self.children[action].as_ref().map(|st| &**st)
    }

    fn state(&self)->&Array1<f32>{
        &self.state
    }

    /* fn parent(&self) ->Option<&RefCell<Self>>{
        self.parent.as_ref().map(|st| &**st)
    } */

    fn player(&self)->i32{
        self.player
    }
 
    fn visits(&self)->usize{
        self.visits
    }

    fn wins(&self)->f32{
        self.wins
    }

    fn untried_actions(&self)->&Vec<usize>{
        &self.untried_actions
    }
}


#[derive(PartialEq, Debug)]
pub enum Result{
    WIN, LOSE, DRAW, INPROGRESS
}
/* {
    state: Array1<f32>,
    parent: MonteCarloSearchTree,


} */
/* impl MonteCarloSearchTree{
    pub fn new(){

    }
    
} */

