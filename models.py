import torch
from torch import nn

# Multilayer perceptron model
class MLP(nn.Module):
    def __init__(self,  num_users, num_items, embedding_dim, layers) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        self.mlp_layers = nn.ModuleList()
        for (in_size, out_size) in zip(layers[:-1], layers[1:]):
            self.mlp_layers.append(nn.Linear(in_size, out_size))

        self.out_layer = nn.Linear(layers[-1], 1)
        # self.sig = nn.Sigmoid()

    def forward(self, userID, itemID):
        user_embedding = self.user_embedding(userID)
        item_embedding = self.item_embedding(itemID)

        out = torch.cat([user_embedding, item_embedding], dim=1)

        for layer_i in range(len(self.mlp_layers)):
            out = self.mlp_layers[layer_i](out)
            out = nn.ReLU()(out)
            # out = nn.Dropout()(out)

        logits = self.out_layer(out)
        
        # rating = self.sig(logits)        
        return logits.view(-1)

# Generalized Matrix Factorization
class GMF(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        self.out_layer = torch.nn.Linear(in_features=embedding_dim, out_features=1)
        # self.logistic = torch.nn.Sigmoid()

    def forward(self, userID, itemID):
        user_embedding = self.embedding_user(userID)
        item_embedding = self.embedding_item(itemID)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.out_layer(element_product)

        return logits.view(-1)

# Neural Matrix Factorization
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers, use_pretrained) -> None:
        super().__init__()    
    

        self.user_embedding_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        self.user_embedding_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)


        self.mlp_layers = nn.ModuleList()
        for (in_size, out_size) in zip(layers[:-1], layers[1:]):
            self.mlp_layers.append(nn.Linear(in_size, out_size))

        self.out_layer = torch.nn.Linear(in_features=layers[-1] + embedding_dim, out_features=1)
        # self.logistic = torch.nn.Sigmoid()
        
        self._init_weight_(use_pretrained)

    def forward(self, userID, itemID):
        user_embedding_mlp = self.user_embedding_mlp(userID)
        item_embedding_mlp = self.item_embedding_mlp(itemID)
        user_embedding_mf = self.user_embedding_gmf(userID)
        item_embedding_mf = self.item_embedding_gmf(itemID)

        mlp_out = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        gmf_out =torch.mul(user_embedding_mf, item_embedding_mf)

        for layer_i in range(len(self.mlp_layers)):
            mlp_out = self.mlp_layers[layer_i](mlp_out)
            mlp_out = nn.ReLU()(mlp_out)

        neuMF_out = torch.cat([mlp_out, gmf_out], dim=-1)
        logits = self.out_layer(neuMF_out)

        return logits.view(-1)

    def _init_weight_(self, use_pretrained):

        if use_pretrained:
            pass
        else:
            nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
            nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
            nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
            nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)

            for layer in self.mlp_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            
            nn.init.kaiming_uniform_(self.out_layer.weight, 
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        
