use image::{ImageReader};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayView3, Axis, NewAxis};


// fn print_shape(a: Array[T, T]) {
//     let shape = a.shape();
//     println!("array shape {}", shape.to_vec());
// }

fn roll1d(a: &ArrayView1<f32>, roll_amount: i32) -> Array1<f32> {
    
    return ndarray::concatenate![
        Axis(0), 
        a.slice(s![-roll_amount..]), 
        a.slice(s![..-roll_amount])]
}


fn roll2d(a: &ArrayView2<f32>, axis: usize, roll_amount: i32) -> Array2<f32> {
    assert!(roll_amount.abs() > 0);
    if axis == 0 {
        return ndarray::concatenate![Axis(0), a.slice(s![-roll_amount.., ..]), a.slice(s![..-roll_amount, ..])]
    } else if axis == 1 {
        ndarray::concatenate![Axis(1), a.slice(s![.., -roll_amount..]), a.slice(s![.., ..-roll_amount,])]
    } else {
        return a.to_owned()
    }
}


// fn gradient(u: &ArrayView2<f32>) -> Array3<f32> {
//     let u_shape = u.shape();
//     let (h, w) = (u_shape[0], u_shape[1]);
//     // let mut grad = Array3::<f32>::zeros((h, w, 2));
//     // grad.slice(s![.., .., 0]) = 1.;
//     // grad
//     let x_shifted_left = &u.slice(s![.., 1..]);
//     let x_shifted_right = &u.slice(s![.., ..-1]);
//     let x_wrap = &u.slice(s![.., 0]) - &u.slice(s![.., -1]);
//     let x_wrap = x_wrap.slice(s![.., NewAxis]);
    
//     let grad_x = x_shifted_left - x_shifted_right;

//     let grad_x = 
//         ndarray::concatenate![Axis(1), grad_x, x_wrap];

//     let y_shifted_left = &u.slice(s![1.., ..]);
//     let y_shifted_right = &u.slice(s![..-1, ..]);
//     let y_wrap = &u.slice(s![0, ..]) - &u.slice(s![-1, ..]);
//     let y_wrap = y_wrap.slice(s![NewAxis, ..]);
    
//     let grad_y = y_shifted_left - y_shifted_right;
//     let grad_y = 
//         ndarray::concatenate![Axis(0), grad_y, y_wrap];

//     ndarray::stack![Axis(2), grad_x, grad_y]        
// }

// fn divergence(p: &ArrayView3<f32>) -> Array3<f32> {
//     let mut div = Array3::<f32>::zeros((32, 32));
//     div
// }

fn gradient(u: &ArrayView2<f32>) -> Array3<f32> {
    let grad_x = roll2d(&u.view(), 1, -1) - u;
    let grad_y = roll2d(&u.view(), 0, -1) - u;

    ndarray::stack![Axis(2), grad_x, grad_y]
}

fn divergence(p: &ArrayView3<f32>) -> Array2<f32> {
    let first_term = p.slice(s![.., .., 0]).to_owned() 
        - roll2d(&p.slice(s![.., .., 0]), 1, 1);
    let second_term = p.slice(s![.., .., 1]).to_owned() 
    - roll2d(&p.slice(s![.., .., 1]), 0, 1);
    -(first_term + second_term)
}



fn main() {
    println!("Hello, world!");

    let array1 = Array2::<f32>::from_shape_vec(
        (3, 5), 
        vec![1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 1., 2., 3., 4., 5.]).unwrap();
    println!("rolled 0 1 {:?}", roll2d(&array1.view(), 0, 1));
    println!("rolled 0 -1 {:?}", roll2d(&array1.view(), 0, -1));
    println!("rolled 1 1 {:?}", roll2d(&array1.view(), 1, 1));
    println!("rolled 1 -1 {:?}", roll2d(&array1.view(), 1, -1));

    let array1 = Array1::<f32>::from_vec(
        (1..10).into_iter()
        .map(|i| i as f32 / 10.).collect()
    );
    println!("rolled {:?}", roll1d(&array1.view(), 1));
    println!("rolled {:?}", roll1d(&array1.view(), 3));
    println!("rolled {:?}", roll1d(&array1.view(), -1));
    println!("rolled {:?}", roll1d(&array1.view(), -3));

    let mut array1 = Array2::<f32>::zeros((10, 10));
    let mut subslice = array1.slice_mut(s![.., 2..]);
    subslice += 0.1;
    let array2 = Array2::<f32>::ones((10, 10));
    let array3 = array2 - array1;
    for (idx, val) in array3.iter().enumerate() {
        println!("value is {} at index {}", val, idx);
    }

    let array3_grad = gradient(&array3.view());
    for (idx, val) in array3_grad.iter().enumerate() {
        println!("grad value is {} at index {}", val, idx)
    }
    println!("grad shape is {:?}", array3_grad.shape());
}
