use rerun;
pub fn display(rec: &rerun::RecordingStream, state: &Vec<i32>, rows: usize, cols:usize, turn: i32){
    rec.set_duration_secs("elapsed", turn);
    let states: Vec<rerun::Color> = state.iter().map(|s|{
        match *s{
            0 => rerun::Color::from_rgb(0, 255, 0),
            1 => rerun::Color::from_rgb(0,0,255),
            2 => rerun::Color::from_rgb(255,0,0),
            _ => panic!("invalid color")
        }
    }).collect();

    let centers = (0..rows).fold(Vec::new(), |mut acc: Vec<(f32,f32)>, row|{
        let cols: Vec<(f32,f32)> = (0..cols).map(|col|{
            (row as f32, col as f32)
        }).collect();
        acc.extend(cols);
        acc
    });
    let _ = rec.log("states", &rerun::Boxes2D::from_centers_and_sizes(centers, vec![(0.9, 0.9); rows*cols]).with_colors(states));
}