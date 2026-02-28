use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::collections::HashMap;

const DEPTH: u8 = 3;
const TREE_COUNT: u8 = 100;
const LEARNING_RATE: f32 = 0.05;

struct Period {
    data: HashMap<String, f32>,
    label: bool,
    residual: f32,
}

#[derive(Debug)]
enum Node {
    Decision {
        indicator: String,
        threshold: f32,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
    },
    Leaf {
        probability: f32,
    },
}


fn load_data(filename: String) -> String {
    // Open the data file
    let path = Path::new(&filename);
    let mut file = match File::open(&path) {
        Err(reason) => panic!("Panic opening training file: {}", reason),
        Ok(file) => file,
    };
    // Read the data file
    let mut data = String::new();
    match file.read_to_string(&mut data) {
        Err(reason) => panic!("Panic reading from training file: {}", reason),
        Ok(bytes) => println!("Read {} bytes of data", bytes),
    };
    data
}

fn main() {
    println!("Hello, world!");
    
    let data: String = load_data("data_small.csv".to_string());

    // Split the data into lines, filtering out empty lines
    let rows: Vec<_> = data.split("\n").filter(|line| !line.is_empty()).collect();
    // The first row - the column titles. Derived here so that I can index this in the loop to get the column titles without deriving over and over
    let columns: Vec<&str> = rows[0].split(",").collect();
    let mut data_points: Vec<Period> = Vec::new();

    // Skip one because the first row is column titles
    let mut total_buys: u32 = 0;
    let mut total_sells: u32 = 0;
    for i in 1..rows.len()-1 {  // Skip header and last row
        // New data point
        let mut new_period = Period { data: HashMap::new(), label: false, residual: 0.0 };
        
        // Get current row's features
        let column_data: Vec<_> = rows[i].split(",").map(|item| item.trim()).collect();
        for (index, column_data_point) in column_data.iter().enumerate() {
            new_period.data.insert(columns[index].to_string(), column_data_point.parse::<f32>().unwrap());
        }
        
        // Get NEXT row's open and close to set label
        let next_row_data: Vec<_> = rows[i+1].split(",").map(|item| item.trim()).collect();
        let next_open: f32 = next_row_data[columns.iter().position(|&c| c == "open").unwrap()].parse().unwrap();
        let next_close: f32 = next_row_data[columns.iter().position(|&c| c == "close").unwrap()].parse().unwrap();
        
        // Label based on next period
        let label: bool;
        if next_close >= next_open {
            label = true;
            total_buys += 1;
        } else {
            label = false;
            total_sells += 1;
        }
        new_period.label = label;
        
        data_points.push(new_period);
    }
    let initial_prediction = total_buys as f32/(total_sells+total_buys) as f32;

    for period in &mut data_points {
        period.residual = period.label as i32 as f32 - initial_prediction;
    }

    let mut trees: Vec<Node> = Vec::new();

    for i in 0..TREE_COUNT {
        println!("Generating tree {}", i);
        let mut new_tree = Node::Decision { indicator: String::new(), threshold: 0.0, left: None, right: None};
        generate_tree(&mut new_tree, &data_points.iter().collect::<Vec<_>>(), 0);
        //  println!("Updating residuals from tree {}", i);
        

        for period in &mut data_points {
            let mut current: &Node = &new_tree;
            loop {
                // println!("Looking at {:?}", current);

                match current {
                    Node::Leaf { probability } => {
                        period.residual += LEARNING_RATE * probability;
                        break;
                    }
                    Node::Decision { indicator, threshold, left, right } => {
                        // println!("Indicator: {}, Threshold: {}", indicator, threshold);
                        if *period.data.get(indicator).unwrap() <= *threshold {
                            current = left.as_ref().unwrap();  // as_ref() gives &Box, auto-derefs to &Node
                        } else {
                            current = right.as_ref().unwrap();
                        }
                    },

                }
            }
        }
        trees.push(new_tree);
    }

    test(trees, initial_prediction);
}

fn test(nodes: Vec<Node>, initial_prediction: f32) {
    let data: String = load_data("test.csv".to_string());
    // Split the data into lines, filtering out empty lines
    let rows: Vec<_> = data.split("\n").filter(|line| !line.is_empty()).collect();
    // The first row - the column titles. Derived here so that I can index this in the loop to get the column titles without deriving over and over
    let columns: Vec<&str> = rows[0].split(",").collect();
    let mut data_points: Vec<Period> = Vec::new();

    // -> count total buys, sells; parses data string to Vec<Period> (and sets correct labels)
    // Todo: use a dynamic array, not a hashmap indexed by strings for the data: Hashmap<String, f32> in Period

    // Skip one because the first row is column titles
    let mut total_buys: u32 = 0;
    let mut total_sells: u32 = 0;
    for i in 1..rows.len()-1 {  // Skip header and last row
        // New data point
        let mut new_period = Period { data: HashMap::new(), label: false, residual: 0.0 };
        
        // Get current row's features
        let column_data: Vec<_> = rows[i].split(",").map(|item| item.trim()).collect();
        for (index, column_data_point) in column_data.iter().enumerate() {
            new_period.data.insert(columns[index].to_string(), column_data_point.parse::<f32>().unwrap());
        }
        
        // Get NEXT row's open and close to set label
        let next_row_data: Vec<_> = rows[i+1].split(",").map(|item| item.trim()).collect();
        let next_open: f32 = next_row_data[columns.iter().position(|&c| c == "open").unwrap()].parse().unwrap();
        let next_close: f32 = next_row_data[columns.iter().position(|&c| c == "close").unwrap()].parse().unwrap();
        
        // Label based on next period
        let label: bool;
        if next_close >= next_open {
            label = true;
            total_buys += 1;
        } else {
            label = false;
            total_sells += 1;
        }
        new_period.label = label;
        
        data_points.push(new_period);
    }
    // let initial_prediction = total_buys as f32/(total_sells+total_buys) as f32;

    for period in &mut data_points {
        period.residual = period.label as i32 as f32 - initial_prediction;
    }
    let mut correct: u32 = 0;
    let mut incorrect: u32 = 0;

    for data_point in data_points {
        let mut prediction = initial_prediction;
        for tree in &nodes {  // Borrow, don't move
            let mut current = tree;
            loop {
                match current {
                    Node::Decision { indicator, threshold, left, right } => {
                        if *data_point.data.get(indicator).unwrap() <= *threshold {
                            current = left.as_ref().unwrap();
                        } else {
                            current = right.as_ref().unwrap();
                        }
                    },
                    Node::Leaf { probability } => {
                        prediction += LEARNING_RATE * probability;
                        break;
                    }
                };
            }
        }

        // Now check if prediction > 0.5 matches data_point.label
        if (prediction > 0.5 && data_point.label) || (prediction <= 0.5 && !data_point.label) {
            correct += 1;
        } else {
            incorrect += 1;
        }

    }
    println!("{} correct, {} incorrect, {}%correct", correct, incorrect, 100.0* correct as f32 / ((correct as f32)+(incorrect as f32)));
}

// Going into here, we have a decision node to set maybe call generate_tree again
fn generate_tree(decision: &mut Node, data: &[&Period], current_depth: u8) {
    // for _i in 0..current_depth {
    //     print!("    ");
    // }
    // println!("Generate_tree has been called");
    // println!("The size of the datset passed was {}", data.len());
    if data.len() == 1 {  // Minimum leaf size
        *decision = Node::Leaf { probability: data[0].residual };
        return;
    }

    if let Node::Decision { indicator, threshold, left, right } = decision {
    
        // *decision = Node::Decision { indicator, threshold, left, right };
        // let mut indicator = decision.indicator;
        // let mut threshold =  decision.threshold;
        // let mut left  =  decision.left;
        // let mut  right = decision.right;
        
        let columns = data[0].data.keys();
        let mut min_variance_per_column = 100000.0;
        let mut ideal_split_per_column = 0.0;
        let mut ideal_column = String::new();

        let mut ideal_left_mean_per_column: f32 = 0.0;
        let mut ideal_right_mean_per_column: f32 = 0.0;

        for column in columns {
            // for _i in 0..current_depth {
            //     print!("  ");
            // }
            // println!("Sifting through column {}", column);

            let mut sorted_data: Vec<&Period> = data.iter().copied().collect();
            sorted_data.sort_by(|a, b| 
                a.data.get(column).unwrap()
                .partial_cmp(b.data.get(column).unwrap())
                .unwrap()
            );
            let mut min_variance: f32 = 1000000.0; // Large number, make sure it gets set in the first iteration
            let mut ideal_split: f32 = 0.0;
            let mut ideal_left_mean: f32 = 0.0;
            let mut ideal_right_mean: f32 = 0.0;

            let mut running_residual_left: f32 = 0.0;
            let mut running_residual_right: f32 = sorted_data.iter().map(|dp| dp.residual).sum();

            let mut running_residual_square_left: f32 = 0.0;
            let mut running_residual_square_right: f32 = sorted_data.iter().map(|dp| dp.residual * dp.residual).sum();

            let mut sorted_data_len: usize = sorted_data.len();
            for i in 0..sorted_data_len - 1 {
                let data_point = sorted_data[i];
                
                running_residual_left += data_point.residual;
                running_residual_right -= data_point.residual;

                running_residual_square_left += data_point.residual * data_point.residual;
                running_residual_square_right -= data_point.residual * data_point.residual;

                let left_count = (i + 1) as f32;
                let right_count = (sorted_data_len - i - 1) as f32;
                
                let mean_left = running_residual_left / left_count;
                let mean_right = running_residual_right / right_count;

                let variance_left = (running_residual_square_left / left_count) - (mean_left * mean_left);
                let variance_right = (running_residual_square_right / right_count) - (mean_right * mean_right);
                let total_variance = (left_count * variance_left + right_count * variance_right) / sorted_data_len as f32;                
                // println!("Total variance:{}", total_variance);
                
                if total_variance < min_variance {
                    min_variance = total_variance;
                    ideal_split = (sorted_data[i].data.get(column).unwrap() + sorted_data[i+1].data.get(column).unwrap()) / 2.0;
                    ideal_left_mean = mean_left;
                    ideal_right_mean = mean_right;
                }
            }

            /*
            for i in 0..sorted_data.len() - 1 {
                //println!("Looking for variance when the split (by {}) is at {}", column, sorted_data[i].data.get(column).unwrap()+1.0);
                let left_side = &sorted_data[..i+1];     // First i+1 elements
                let right_side = &sorted_data[i+1..];    // Remaining elements
                
                let left_side_mean: f32 = left_side.iter().map(|dp| dp.residual).sum::<f32>() / left_side.len() as f32;
                let right_side_mean: f32 = right_side.iter().map(|dp| dp.residual).sum::<f32>() / right_side.len() as f32;

                let left_side_variance: f32 = left_side.iter().map(|dp| (dp.residual-left_side_mean).powf(2.0)).sum::<f32>() as f32 / left_side.len() as f32;
                let right_side_variance: f32 = right_side.iter().map(|dp| (dp.residual-right_side_mean).powf(2.0)).sum::<f32>() as f32 / right_side.len() as f32;
                let total_variance: f32 = (left_side.len() as f32 * left_side_variance + right_side.len() as f32 * right_side_variance) as f32 / sorted_data.len() as f32;

                if min_variance > total_variance {
                    min_variance = total_variance;
                    ideal_split = (sorted_data[i].data.get(column).unwrap() + sorted_data[i+1].data.get(column).unwrap()) / 2.0;
                    ideal_left_mean = left_side_mean;
                    ideal_right_mean = right_side_mean;
                }
            }            
            */


            if min_variance < min_variance_per_column {
                min_variance_per_column = min_variance;
                ideal_split_per_column = ideal_split;
                ideal_column = column.clone();
                ideal_left_mean_per_column = ideal_left_mean;
                ideal_right_mean_per_column = ideal_right_mean;
            }
        }
        // for _i in 0..current_depth {
        //    // print!("    ");
        // }
        // println!("The ideal variance column was {}, with threshold {}", ideal_column, ideal_split_per_column);
        *indicator = ideal_column.clone();
        *threshold = ideal_split_per_column;
        
        if current_depth + 1 == DEPTH {
            *left = Some(Box::new(Node::Leaf { probability: ideal_left_mean_per_column }));
            *right = Some(Box::new(Node::Leaf { probability: ideal_right_mean_per_column }));
            return;
        }


        let left_data: Vec<&Period> = data.iter()
            .copied()
            .filter(|dp| dp.data.get(indicator).unwrap() <= threshold)
            .collect();

        let right_data: Vec<&Period> = data.iter()
            .copied()
            .filter(|dp| dp.data.get(indicator).unwrap() > threshold)
            .collect();
        
        if left_data.len() <= 1 || right_data.len() <= 1 {
            // for _i in 0..current_depth {
            //   //  print!("    ");
            // }
            // println!("Somehow, one is empty but the indicator did not fire. THis means that the boudnary was right at the edge. ");
            // println!("creating a leaf - no split is optimal.");
       //     println!("leaf");
            // Calculate the mean of all data as the leaf value
            let mean_residual = data.iter()
                .map(|dp| dp.residual)
                .sum::<f32>() / data.len() as f32;
            
            // Replace the current Decision node with a Leaf
            *decision = Node::Leaf { probability: mean_residual };
            return;
        }

        *left = Some(Box::new(Node::Decision { indicator: String::new(), threshold: 0.0, left: None, right: None }));
        *right = Some(Box::new(Node::Decision { indicator: String::new(), threshold: 0.0, left: None, right: None }));
        generate_tree(&mut (*left).as_mut().unwrap(), &left_data, current_depth+1);
        generate_tree(&mut (*right).as_mut().unwrap(), &right_data, current_depth+1);

    
    }
}
