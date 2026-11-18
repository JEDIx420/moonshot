//! Independent CPU-only Cycle 6 artifact verifier. No Burn, CUDA or Python.
//! Usage: cycle6-verify /path/to/cycle6-.../seed_6109

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, env, error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn read(path: &Path) -> Result<Value> { Ok(serde_json::from_slice(&fs::read(path)?)?) }
fn array<'a>(value: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    value[field].as_array().ok_or_else(|| format!("missing array {field}").into())
}
fn number(value: &Value) -> Result<f64> {
    value.as_f64().ok_or_else(|| "expected finite numeric score".into())
}
fn boolean(value: &Value) -> Result<bool> {
    value.as_bool().ok_or_else(|| "expected boolean".into())
}
fn name(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| "expected string".into())
}
fn num(value: &Value) -> Result<usize> {
    Ok(value.as_u64().ok_or("expected unsigned integer")? as usize)
}
fn tokens(value: &Value) -> Result<Vec<usize>> {
    value.as_array().ok_or("expected tokens array")?
        .iter().map(num).collect()
}
fn require(ok: bool, msg: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(msg.to_string().into()) }
}
fn approx(a: f64, b: f64, desc: &str) -> Result<()> {
    require(a.is_finite() && b.is_finite() && (a-b).abs() < 1e-10,
            &format!("score disagreement in {desc}: {a} != {b}"))
}
fn digest(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn check_checkpoint(root: &Path, info: &Value, desc: &str) -> Result<()> {
    let file = name(&info["file"])?;
    require(file == "model.safetensors", "unexpected checkpoint filename")?;
    let actual = digest(&root.join(file))?;
    require(actual == name(&info["sha256"])? && boolean(&info["cold_reload_predicted_same"])? &&
            num(&info["parameters"])? == 1_095_744,
            &format!("bad checkpoint hash/reload claim: {desc}"))
}
fn transform(op: usize, [a,b,c]: [usize;3]) -> [usize;3] {
    match op {
        0 => [b,c,a], 1 => [c,a,b],
        2 => [(a+1)%8,(b+1)%8,(c+1)%8],
        3 => [(a+2)%8,b,c], 4 => [c,b,a],
        5 => [a^b,b^c,c^a],
        6 => [(a+b)%8,(b+c)%8,(c+a)%8],
        7 => if a%2==0 {[a,c,b]} else {[a,b,c]},
        8 => [a,(b+a)%8,(c+a)%8],
        9 => [a,a^b,b^c],
        _ => panic!("invalid operator"),
    }
}
fn check_example(example: &Value) -> Result<Vec<usize>> {
    let input = tokens(&example["input_tokens"])?;
    let expected = tokens(&example["target_tokens"])?;
    require(input.len()==7 && expected.len()==4 && input[0]==1 && input[6]==3 && expected[3]==2,
            "invalid Cycle6 grammar")?;
    require((20..30).contains(&input[1]) && (input[2]==30 || (20..30).contains(&input[2]))
        && input[3..6].iter().all(|&t| (4..12).contains(&t)),
        "invalid operator or symbol token")?;
    let state = [input[3]-4,input[4]-4,input[5]-4];
    let first = transform(input[1]-20,state);
    let result = if input[2]==30 { first } else { transform(input[2]-20,first) };
    require(expected == vec![4+result[0],4+result[1],4+result[2],2],
            "target label disagrees with independent oracle")?;
    Ok(input)
}
fn score(examples: &[Value], predictions: &[Value], desc: &str) -> Result<f64> {
    require(examples.len()==predictions.len() && !examples.is_empty(),
            &format!("missing predictions in {desc}"))?;
    let mut correct = 0usize;
    for (example,prediction) in examples.iter().zip(predictions) {
        check_example(example)?;
        if tokens(prediction)? == tokens(&example["target_tokens"])? { correct+=1; }
    }
    Ok(correct as f64/examples.len() as f64)
}
fn mean(numbers: &[f64]) -> f64 {
    if numbers.is_empty(){0.0}else{numbers.iter().sum::<f64>()/numbers.len() as f64}
}
fn verify(root: &Path) -> Result<()> {
    let metadata=read(&root.join("cycle6_run.json"))?;
    require(name(&metadata["status"])?=="completed" && name(&metadata["backend"])?=="burn-cuda",
            "run did not complete CUDA training")?;
    let primitive_path = root.join("primitive_splits.json");
    let pair_path = root.join("operator_pair_splits.json");
    require(name(&metadata["dataset_sha256"]["primitive_splits"])?==digest(&primitive_path)? &&
            name(&metadata["dataset_sha256"]["operator_pair_splits"])?==digest(&pair_path)?,
            "frozen dataset SHA mismatch")?;
    let primitive=read(&primitive_path)?;
    let tasks=primitive.as_array().ok_or("bad primitive dataset")?;
    require(tasks.len()==10,"not 10 primitive tasks")?;
    for (op,task) in tasks.iter().enumerate() {
        let mut all=HashSet::new();
        for (field,n) in [("train",384),("validation",64),("test",64)] {
            let values=array(task,field)?;
            require(values.len()==n,"wrong primitive split size")?;
            for example in values {
                let ids=check_example(example)?;
                require(ids[1]==20+op && ids[2]==30,"incorrect primitive operator")?;
                require(all.insert(ids),"primitive train/val/test overlap")?;
            }
        }
        require(all.len()==512,"incomplete primitive domain")?;
    }
    let pairs=read(&pair_path)?;
    let groups=pairs.as_array().ok_or("bad pair groups")?;
    require(groups.len()==88,"incorrect ordered-pair population")?;
    let mut ids=HashSet::new();
    let mut pair_counts=[0usize;3];
    for group in groups {
        let a=num(&group["a"])?;
        let b=num(&group["b"])?;
        require(a!=b && a<10 && b<10 && ids.insert((a,b)), "bad/duplicate pair")?;
        let role=name(&group["role"])?;
        let (role_idx,field) = match role {"train"=>(0,"training"),"validation"=>(1,"evaluation"),
                                       "test"=>(2,"evaluation"),_=>return Err("bad pair role".into())};
        pair_counts[role_idx]+=1;
        let examples=array(group,field)?;
        require(examples.len()==64,"wrong pair split size")?;
        let mut unique=HashSet::new();
        for example in examples {
            let input=check_example(example)?;
            require(input[1]==20+a && input[2]==20+b && unique.insert(input),
                    "wrong pair target/operator or duplicate")?;
        }
        let supports=array(group,"support")?;
        if role=="test" {
            require(supports.len()==16,"wrong 16-shot support set")?;
            for support in supports {
                let input=check_example(support)?;
                require(!unique.contains(&input), "fewshot support leaked into test")?;
            }
        } else { require(supports.is_empty(),"support data on non-test pair")?; }
    }
    require(pair_counts==[48,20,20],"wrong pair partition")?;
    let heldout_pairs:Vec<_>=groups.iter().filter(|p|p["role"].as_str()==Some("test")).collect();

    let seq_dir=root.join("sequential");
    let seq=read(&seq_dir.join("sequential_result.json"))?;
    let acquired=array(&seq,"acquired_task_ids")?;
    let isolated=array(&seq,"isolated")?;
    let isolated_scores=array(&seq,"isolated_scores")?;
    require(isolated.len()==10 && isolated_scores.len()==10,"missing isolated scores")?;
    let isolated_predictions=read(&seq_dir.join("isolated_predictions.json"))?;
    let iso_predictions=isolated_predictions.as_array().ok_or("bad isolated predictions")?;
    require(iso_predictions.len()==10,"missing isolated predictions")?;
    for op in 0..10 {
        check_checkpoint(&seq_dir.join(format!("isolated_{op}")),
                         &isolated[op]["checkpoint"],"isolated")?;
        let values=array(&iso_predictions[op],"input_target")?;
        let predicted=array(&iso_predictions[op],"predictions")?;
        approx(score(values,predicted,"isolated")?,number(&isolated_scores[op])?,"isolated")?;
        approx(number(&isolated[op]["test_accuracy"])?,number(&isolated_scores[op])?,"iso metric")?;
    }
    let actual_acquired:Vec<usize>=isolated_scores.iter().enumerate()
        .filter(|(_,v)|v.as_f64().unwrap_or(0.0)>=0.8).map(|(i,_)|i).collect();
    require(acquired.iter().map(num).collect::<Result<Vec<_>>>()?==actual_acquired,
            "posthoc acquired-task set")?;
    require(boolean(&seq["acquisition_gate"])?==(actual_acquired.len()>=7),"acquisition gate incorrect")?;
    let conditions=&seq["conditions"];
    let mut means=std::collections::BTreeMap::new();
    for condition in ["shared","uniform","feedback"] {
        let metrics=&conditions[condition];
        check_checkpoint(&seq_dir.join(format!("{condition}_final")),
                         &metrics["checkpoint"],condition)?;
        require(array(metrics,"retention_matrix")?.len()==10,"incomplete retention matrix")?;
        let history=read(&seq_dir.join(format!("{condition}_trajectory_predictions.json")))?;
        let trajectory=history.as_array().ok_or("missing trajectory predictions")?;
        require(trajectory.len()==55,"retention trajectory must have 55 task-generation points")?;
        let mut history_ids=HashSet::new();
        for record in trajectory {
            let generation=num(&record["generation"])?;
            let task=num(&record["task"])?;
            require((1..=10).contains(&generation) && task<generation &&
                    history_ids.insert((generation,task)),"duplicate/invalid retention row")?;
            let examples=array(record,"examples")?;
            require(examples==array(&tasks[task],"test")?,"trajectory test data mismatch")?;
            let accuracy=score(examples,array(record,"predictions")?,"retention history")?;
            approx(accuracy,number(&record["exact_match"])?,"trajectory record")?;
            approx(accuracy,number(&metrics["retention_matrix"][generation-1][task])?,
                   "retention matrix cell")?;
        }
        let records=read(&seq_dir.join(format!("{condition}_final_predictions.json")))?;
        let items=records.as_array().ok_or("bad final predictions")?;
        require(items.len()==640,"final predictions missing 640 rows")?;
        let mut scores=Vec::new();
        for op in 0..10 {
            let slice=&items[op*64..(op+1)*64];
            let test=&array(&tasks[op],"test")?;
            let mut correct=0usize;
            for (record,example) in slice.iter().zip(test.iter()) {
                require(tokens(&record["input"])?==tokens(&example["input_tokens"])? &&
                        tokens(&record["target"])?==tokens(&example["target_tokens"])? &&
                        boolean(&record["correct"])?==
                            (tokens(&record["predicted"])?==tokens(&record["target"])?),
                        "final prediction row mismatch")?;
                if tokens(&record["predicted"])?==tokens(&example["target_tokens"])? {correct+=1;}
            }
            let accuracy=correct as f64/64.0;
            approx(accuracy,number(&metrics["final_test_accuracy"][op])?,condition)?;
            scores.push(accuracy);
        }
        means.insert(condition,mean(&actual_acquired.iter().copied().filter(|&i|i<9)
            .map(|i|scores[i]).collect::<Vec<_>>()));
    }
    let feedback=means["feedback"];
    let uniform=means["uniform"];
    let shared=means["shared"];
    let last=number(&conditions["feedback"]["final_test_accuracy"][9])?;
    let acquired_at_birth=actual_acquired.iter().all(|&task| {
        conditions["feedback"]["retention_matrix"][task][task].as_f64().unwrap_or(0.0)>=0.80
    });
    let retained=actual_acquired.len()>=7 && feedback>=0.70 && feedback-uniform>=0.10
        && feedback-shared>=0.20 && acquired_at_birth && last>=0.80;
    require(boolean(&seq["retention_gate"])?==retained,"retention gate disagrees with predictions")?;

    let comp_dir=root.join("composition");
    let comp=read(&comp_dir.join("composition_result.json"))?;
    let mut zero=std::collections::BTreeMap::new();
    for condition in ["primitives_only","pairs_and_primitives"] {
        let info=&comp["pretraining"][condition];
        check_checkpoint(&comp_dir.join(condition),&info["checkpoint"],condition)?;
        let records=read(&comp_dir.join(format!("{condition}_heldout_predictions.json")))?;
        let record=records.as_array().ok_or("bad paired zero-shot predictions")?;
        require(record.len()==20,"not twenty heldout pairs")?;
        let mut values=Vec::new();
        for (index,item) in record.iter().enumerate() {
            let examples=array(item,"test")?;
            let predictions=array(item,"predictions")?;
            require(item["pair"]==serde_json::json!([
                        heldout_pairs[index]["a"],heldout_pairs[index]["b"]]) &&
                    examples==array(heldout_pairs[index],"evaluation")?,
                    "zero-shot evaluated a different/seen pair or changed its test cases")?;
            let acc=score(examples,predictions,"zero-shot pair")?;
            approx(acc,number(&item["exact_match"])?,"zero-shot stored score")?;
            values.push(acc);
        }
        approx(mean(&values),number(&info["heldout_pair_mean"])?,"zero-shot mean")?;
        zero.insert(condition,mean(&values));
        let source_records=read(&comp_dir.join(format!("{condition}_primitive_predictions.json")))?;
        let source=source_records.as_array().ok_or("missing source primitive predictions")?;
        require(source.len()==10,"source primitive scoring incomplete")?;
        let mut acquired=0usize;
        for (op,record) in source.iter().enumerate() {
            require(num(&record["task"])?==op &&
                    array(record,"test")?==array(&tasks[op],"test")?,
                    "source primitive test split disagreement")?;
            let accuracy=score(array(record,"test")?,array(record,"predictions")?,"source task")?;
            approx(accuracy,number(&record["exact_match"])?,"source task stored score")?;
            approx(accuracy,number(&info["primitive_test_scores"][op])?,"source task summary")?;
            if accuracy>=0.80 {acquired+=1;}
        }
        require(acquired==num(&info["primitive_skills_acquired"])? &&
                acquired<=10,"source skill count is inaccurate")?;
    }
    let mut shots=std::collections::BTreeMap::new();
    for condition in ["random_init","primitives_only","pairs_and_primitives"] {
        let records=read(&comp_dir.join(format!("fewshot_{condition}_predictions.json")))?;
        let results=records.as_array().ok_or("bad fewshot predictions")?;
        require(results.len()==20,"missing fewshot pair")?;
        let mut values=Vec::new();
        for (index,item) in results.iter().enumerate() {
            let support=array(item,"support")?;
            let examples=array(item,"test")?;
            require(item["pair"]==serde_json::json!([
                        heldout_pairs[index]["a"],heldout_pairs[index]["b"]]) &&
                    support==array(heldout_pairs[index],"support")? &&
                    examples==array(heldout_pairs[index],"evaluation")?,
                    "fewshot test/support set differs from preregistered heldout pair")?;
            require(support.len()==16,"fewshot support wrong size")?;
            let test_ids:HashSet<_>=examples.iter().map(|e|tokens(&e["input_tokens"]).unwrap()).collect();
            for ex in support {require(!test_ids.contains(&check_example(ex)?),"fewshot support leaked")?;}
            let acc=score(examples,array(item,"predictions")?,"fewshot")?;
            approx(acc,number(&item["exact_match"])?,"fewshot stored score")?;
            values.push(acc);
        }
        approx(mean(&values),number(&comp["fewshot"][condition]["mean_exact_match"])?,
               "fewshot mean")?;
        shots.insert(condition,mean(&values));
    }
    let baseline=&comp["no_learning_baselines"];
    let joint=zero["pairs_and_primitives"];
    let primitive=zero["primitives_only"];
    let copied=number(&baseline["identity_copy_mean"])?;
    let first=number(&baseline["first_operator_mean"])?;
    let mut copied_recomputed=Vec::new();
    let mut first_recomputed=Vec::new();
    for (index,group) in heldout_pairs.iter().enumerate() {
        let a=num(&group["a"])?;
        let examples=array(group,"evaluation")?;
        let mut copy_correct=0;
        let mut first_correct=0;
        for example in examples {
            let input=tokens(&example["input_tokens"])?;
            let target=tokens(&example["target_tokens"])?;
            let state=[input[3]-4,input[4]-4,input[5]-4];
            let one=transform(a,state);
            if target==vec![state[0]+4,state[1]+4,state[2]+4,2] {copy_correct+=1;}
            if target==vec![one[0]+4,one[1]+4,one[2]+4,2] {first_correct+=1;}
        }
        let c=copy_correct as f64/examples.len() as f64;
        let f=first_correct as f64/examples.len() as f64;
        approx(c,number(&baseline["identity_copy_per_pair"][index])?,"copy baseline")?;
        approx(f,number(&baseline["first_operator_per_pair"][index])?,"one-op baseline")?;
        copied_recomputed.push(c);
        first_recomputed.push(f);
    }
    approx(copied,mean(&copied_recomputed),"copy baseline overall")?;
    approx(first,mean(&first_recomputed),"one-op baseline overall")?;
    let source=num(&comp["pretraining"]["pairs_and_primitives"]["primitive_skills_acquired"])?;
    let transfer=source>=7 && joint>=0.50 && joint-primitive>=0.20 &&
        joint-copied>=0.20 && joint-first>=0.20 &&
        shots["pairs_and_primitives"]-shots["primitives_only"]>=0.10;
    require(boolean(&comp["composition_gate"])?==transfer,"composition gate disagrees with scores")?;
    require(boolean(&metadata["scientific_leap_gate"])?==
            (actual_acquired.len()>=7 && retained && transfer),"final gate mismatch")?;
    println!("[AUDIT PASS] seed={} acquired={} retention={} composition={} joint_ood={:.4}",
             num(&metadata["seed"])?,actual_acquired.len(),retained,transfer,joint);
    Ok(())
}
fn main() {
    let args:Vec<_>=env::args().collect();
    if args.len()!=2 {
        eprintln!("Usage: cycle6-verify /path/to/cycle6-run/seed_6109");
        std::process::exit(2);
    }
    if let Err(error)=verify(Path::new(&args[1])) {
        eprintln!("[AUDIT FAIL] {error}");
        std::process::exit(1);
    }
}
