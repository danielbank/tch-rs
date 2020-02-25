extern crate tch;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub fn train(vs: &mut nn::VarStore) -> failure::Fallible<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..10 {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}

fn main() -> failure::Fallible<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut vs = nn::VarStore::new(Device::Cpu);
    if args.len() < 2 {
        train(&mut vs)?;
        vs.save("weights.pt")?;
    } else {
        let _ = net(&vs.root());
        vs.load(args[1].as_str())?;
    }

    println!("{:#?}", vs.root());
    println!("{:#?}", Vec::<f64>::from(&vs.root().get("bias").unwrap()));
    Ok(())
}
