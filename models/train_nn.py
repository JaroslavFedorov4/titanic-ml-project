import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from preprocessing.preprocess import prep_NN
from config import CATEGORICAL_FEATURES, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE
from models.models_nn import TitanicDATASET, TitanicMLP

def train_loop(model, train_loader, loss_fn, optim):
    model.train()

    total_loss = 0

    for X_num_batch, X_cat_batch, y_batch in train_loader:

        X_num_batch = X_num_batch.to(DEVICE)
        X_cat_batch = X_cat_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        pred = model(X_num_batch, X_cat_batch)
        loss = loss_fn(pred, y_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)
    
def eval_loop(model, val_loader, loss_fn):
    model.eval()

    total_loss = 0

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in val_loader:

            X_num_batch = X_num_batch.to(DEVICE)
            X_cat_batch = X_cat_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_num_batch, X_cat_batch)
            loss = loss_fn(pred, y_batch)

            probs = torch.softmax(pred, dim=1)[:, 1]
            preds = torch.argmax(pred, dim=1)

            total_loss += loss.item()

            all_preds.extend(preds.detach().cpu().numpy().ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())
            all_targets.extend(y_batch.detach().cpu().numpy().ravel())

        avg_loss = total_loss / len(val_loader)

        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy
    
def train_nn(X_train, X_val, y_train, y_val, numeric_features, categorical_features, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    num_preprocessor, cat_preprocessor = prep_NN(numeric_features, categorical_features)

    X_train_num = num_preprocessor.fit_transform(X_train)
    X_val_num = num_preprocessor.transform(X_val)

    X_train_cat = cat_preprocessor.fit_transform(X_train)
    X_val_cat = cat_preprocessor.transform(X_val)

    X_train_cat = X_train_cat.astype(int) + 1
    X_val_cat = X_val_cat.astype(int) + 1

    X_train_num_NN = torch.tensor(X_train_num, dtype=torch.float32)
    X_val_num_NN = torch.tensor(X_val_num, dtype=torch.float32)

    X_train_cat_NN = torch.tensor(X_train_cat, dtype=torch.long)
    X_val_cat_NN = torch.tensor(X_val_cat, dtype=torch.long)

    y_train_NN = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_val_NN = torch.tensor(y_val.to_numpy(), dtype=torch.long)

    cat_sizes = np.maximum(
        X_train_cat.max(axis=0),
        X_val_cat.max(axis=0)
    ) + 1

    embedding_size = []

    for val in cat_sizes:
        val = int(val)
        if val > 50:
            embedding_size.append((val, 8))
        else:
            embedding_size.append((val, 2))

    train_dataset = TitanicDATASET(X_train_num_NN, X_train_cat_NN, y_train_NN)
    val_dataset = TitanicDATASET(X_val_num_NN, X_val_cat_NN, y_val_NN)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train_num_NN.shape[1]
    model = TitanicMLP(input_size, embedding_size).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0
    stop_epoch = 10
    train_loss_mass = []
    vall_loss_mass = []


    for epoch in range(epochs):
        train_loss = train_loop(model, train_loader, loss_fn, optimizer)

        eval_loss, eval_acc = eval_loop(model, val_loader, loss_fn)

        scheduler.step(eval_loss)

        train_loss_mass.append(train_loss)
        vall_loss_mass.append(eval_loss)

        print(f"Epoch: {epoch}, train_loss: {train_loss}, eval_loss: {eval_loss}, eval_rmse: {eval_acc}")

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            best_val_acc = eval_acc
            best_train_loss = train_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        if counter >= stop_epoch:
            print("Early Stopping")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    result = {
        "model": "NeuralNetwork",
        "best_score": best_val_acc,
        "best_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "model_architecture": "MLP + embedding"
        },
        "train_loss": best_train_loss,
        "model_object": model
    }

    return result