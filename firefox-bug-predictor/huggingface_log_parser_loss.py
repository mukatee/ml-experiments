import json

#this parses the huggingface training script output to list evaluation accuracy and loss evolution over epochs,
#collecting the row with the lowest loss

target_epoch = 3.0

#with open("huggingface-train-5-epochs.txt", "r") as f:
#with open("huggingface-train-3-epochs-notriage.txt", "r") as f:
with open("huggingface-train-1.txt", "r") as f:
    log_text = f.read()

iterations = 0
best_loss = 100
best_epoch = 0
best_accuracy = 0
for line in log_text.splitlines():
    if line.startswith("{'eval_loss':"):
        line = line.replace("'", '"')
        line_json = json.loads(line)
        #print(line_json["eval_loss"])
        epoch = float(line_json["epoch"])
        eval_loss = float(line_json["eval_loss"])
        accuracy = float(line_json["eval_accuracy"])
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_accuracy = accuracy
        if epoch == target_epoch:
            iterations += 1
            print(f"iteration {iterations}: best_loss={best_loss}, best_epoch={best_epoch}, best_accuracy={best_accuracy}")
            best_loss = 100
            best_epoch = 0
