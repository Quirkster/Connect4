use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use ndarray::{Array1, Array2};

use crate::neuralnetwork::LinearLayer; // Use little endian for portability

pub fn save_layers(path: &str, layers: &[LinearLayer]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_u32::<LittleEndian>(layers.len() as u32)?;

    for layer in layers {
        write_array2(&mut writer, &layer.weights)?;
        write_array1(&mut writer, &layer.biases)?;
        println!("{:?}", layer.weight_grads);
        write_array2(&mut writer, &layer.weight_grads)?;
        write_array1(&mut writer, &layer.bias_grads)?;
    }

    Ok(())
}

fn write_array2<W: Write>(writer: &mut W, array: &Array2<f32>) -> std::io::Result<()> {
    let (rows, cols) = array.dim();
    writer.write_u32::<LittleEndian>(rows as u32)?;
    writer.write_u32::<LittleEndian>(cols as u32)?;

    /* let data = array.as_slice().expect("Array2 not contiguous");
    for &val in data {
        writer.write_f32::<LittleEndian>(val)?;
    } */

   let raw = array.to_owned().into_raw_vec(); // Guaranteed to be flat and correct
    for val in raw {
        writer.write_f32::<LittleEndian>(val)?;
    }
    Ok(())
}

fn write_array1<W: Write>(writer: &mut W, array: &Array1<f32>) -> std::io::Result<()> {
    writer.write_u32::<LittleEndian>(array.len() as u32)?;
    let data = array.as_slice().expect("Array1 not contiguous");
    for &val in data {
        writer.write_f32::<LittleEndian>(val)?;
    }
    Ok(())
}

pub fn load_layers(path: &str) -> std::io::Result<Vec<LinearLayer>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    println!("reader!");
    let num_layers = reader.read_u32::<LittleEndian>()? as usize;
    println!("num_layers: {num_layers}");
    let mut layers = Vec::with_capacity(num_layers);
    println!("layers!");
    for _ in 0..num_layers {
        let weights = read_array2(&mut reader)?;
        let biases = read_array1(&mut reader)?;
        let weight_grads = read_array2(&mut reader)?;
        let bias_grads = read_array1(&mut reader)?;
        layers.push(LinearLayer {
            weights,
            biases,
            weight_grads,
            bias_grads,
        });
    }

    Ok(layers)
}

fn read_array2<R: Read>(reader: &mut R) -> std::io::Result<Array2<f32>> {
    let rows = reader.read_u32::<LittleEndian>()? as usize;
    let cols = reader.read_u32::<LittleEndian>()? as usize;

    let mut buf = vec![0.0f32; rows * cols];
    for i in 0..buf.len() {
        buf[i] = reader.read_f32::<LittleEndian>()?;
    }

    Ok(Array2::from_shape_vec((rows, cols), buf).expect("Invalid shape"))
}

fn read_array1<R: Read>(reader: &mut R) -> std::io::Result<Array1<f32>> {
    let len = reader.read_u32::<LittleEndian>()? as usize;

    let mut buf = vec![0.0f32; len];
    for i in 0..len {
        buf[i] = reader.read_f32::<LittleEndian>()?;
    }

    Ok(Array1::from_vec(buf))
}
