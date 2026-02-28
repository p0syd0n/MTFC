use std::fs::File;
use std::io::prelude::*;

use crate::Node;
use crate::Period;
use crate::GRAPHICAL;
use crate::LOG_FILE;
use crate::DEPTH;
use crate::TREE_COUNT;
use crate::LEARNING_RATE;

pub fn node_to_json(node: &Node) -> String {
    match node {
        Node::Leaf { probability } => {
            format!(r#"{{"type":"leaf","probability":{}}}"#, probability)
        }
        Node::Decision { indicator, threshold, left, right } => {
            let left_json = match left {
                Some(n) => node_to_json(n),
                None => "null".to_string(),
            };
            let right_json = match right {
                Some(n) => node_to_json(n),
                None => "null".to_string(),
            };
            format!(
                r#"{{"type":"decision","indicator":"{}","threshold":{},"left":{},"right":{}}}"#,
                indicator, threshold, left_json, right_json
            )
        }
    }
}

pub fn residual_stats(data: &[Period]) -> (f32, f32) {
    let mean = data.iter().map(|p| p.residual).sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|p| (p.residual - mean).powi(2)).sum::<f32>() / data.len() as f32;
    (mean, variance)
}

pub fn open_log_file() -> std::fs::File {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .expect("Could not open log file")
}

pub fn log_tree_step(tree_index: u8, tree: &Node, data: &[Period]) {
    if !GRAPHICAL { return; }
    let (mean, variance) = residual_stats(data);
    let tree_json = node_to_json(tree);
    let entry = format!(
        "{{\"event\":\"tree\",\"tree_index\":{},\"mean_residual\":{},\"residual_variance\":{},\"tree\":{}}}\n",
        tree_index, mean, variance, tree_json
    );
    let mut f = open_log_file();
    f.write_all(entry.as_bytes()).expect("Could not write tree log entry");
}

pub fn log_test_result(correct: u32, incorrect: u32) {
    if !GRAPHICAL { return; }
    let accuracy = 100.0 * correct as f32 / (correct as f32 + incorrect as f32);
    let entry = format!(
        "{{\"event\":\"test\",\"correct\":{},\"incorrect\":{},\"accuracy\":{}}}\n",
        correct, incorrect, accuracy
    );
    let mut f = open_log_file();
    f.write_all(entry.as_bytes()).expect("Could not write test log entry");
}

pub fn init_log(initial_prediction: f32) {
    if !GRAPHICAL { return; }
    let mut f = File::create(LOG_FILE).expect("Could not create log file");
    let entry = format!(
        "{{\"event\":\"init\",\"depth\":{},\"tree_count\":{},\"learning_rate\":{},\"initial_prediction\":{}}}\n",
        DEPTH, TREE_COUNT, LEARNING_RATE, initial_prediction
    );
    f.write_all(entry.as_bytes()).expect("Could not write init log entry");
}