# import argparse
# import copy
# import pandas as pd
# import os

# from tree_sitter import Language, Parser
# from prepare_data import get_root_paths


# def parse_options():
#     parser = argparse.ArgumentParser(description="TrVD preprocess~.")
#     parser.add_argument(
#         "-i",
#         "--input",
#         default="mutrvd",
#         choices="mutrvd",
#         help="training dataset type",
#         type=str,
#         required=False,
#     )
#     args = parser.parse_args()
#     return args


# def parse_ast(source):
#     CPP_LANGUAGE = Language("build_languages/my-languages.so", "cpp")
#     parser = Parser()
#     parser.set_language(CPP_LANGUAGE)
#     tree = parser.parse(source.encode("utf-8"))
#     return tree


# def parse_ast_js(source):
#     JS_LANGUAGE = Language("build_languages/my-languages.so", "javascript")
#     parser = Parser()
#     parser.set_language(JS_LANGUAGE)
#     tree = parser.parse(source.encode("utf-8"))
#     if not tree or not tree.root_node:
#         print("Empty tree found for source:", source)
#     return tree


# args = parse_options()


# class Pipeline:
#     def __init__(self):
#         self.train = None
#         self.train_keep = None
#         self.train_block = None
#         self.dev = None
#         self.dev_keep = None
#         self.dev_block = None
#         self.test = None
#         self.test_keep = None
#         self.test_block = None
#         self.size = None
#         self.w2v_path = None

#     def parse_source(self, dataset):
#         train = pd.read_pickle("dataset/" + dataset + "/train.pkl")
#         test = pd.read_pickle("dataset/" + dataset + "/test.pkl")
#         dev = pd.read_pickle("dataset/" + dataset + "/val.pkl")

#         print("Number of rows in train dataset:", len(train))
#         print("Number of rows in test dataset:", len(test))
#         print("Number of rows in dev dataset:", len(dev))

#         train["code"] = train["code"].apply(parse_ast_js)
#         self.train = train
#         self.train_keep = copy.deepcopy(train)
#         dev["code"] = dev["code"].apply(parse_ast_js)
#         self.dev = dev
#         self.dev_keep = copy.deepcopy(dev)
#         test["code"] = test["code"].apply(parse_ast_js)
#         self.test = test
#         self.test_keep = copy.deepcopy(test)

#     def dictionary_and_embedding(self, size):
#         self.size = size
#         trees = self.train
#         self.w2v_path = "subtrees/" + args.input + "/node_w2v_" + str(size)
#         if not os.path.exists("subtrees"):
#             os.mkdir("subtrees")
#         if not os.path.exists("subtrees/" + args.input):
#             os.mkdir("subtrees/" + args.input)
#         from prepare_data import get_sequences

#         def trans_to_sequences(ast):
#             sequence = []
#             get_sequences(ast, sequence)

#             if not sequence:
#                 print("Empty sequence found for AST:", ast)

#             paths = []
#             get_root_paths(ast, paths, [])

#             if not paths:
#                 print("Empty paths found for AST:", ast)

#             paths.append(sequence)

#             if not paths:
#                 print("Empty paths after appending sequence for AST:", ast)
#             return paths

#         if not os.path.exists(self.w2v_path):
#             corpus = trees["code"].apply(trans_to_sequences)
#             paths = []
#             for all_paths in corpus:
#                 for path in all_paths:
#                     path = [
#                         token.decode("utf-8") if type(token) is bytes else token
#                         for token in path
#                     ]
#                     paths.append(path)
#             corpus = paths
#             from gensim.models.word2vec import Word2Vec

#             print("corpus size: ", len(corpus))
#             w2v = Word2Vec(corpus, vector_size=size, workers=96, sg=1, min_count=3)
#             print("word2vec : ", w2v)
#             w2v.save(self.w2v_path)

#     def generate_block_seqs_time(self, data):
#         from prepare_data import get_blocks as func
#         from gensim.models.word2vec import Word2Vec

#         word2vec = Word2Vec.load("subtrees/trvd/node_w2v_128").wv
#         max_token = word2vec.vectors.shape[0]

#         def tree_to_index(node):
#             token = node.token
#             if type(token) is bytes:
#                 token = token.decode("utf-8")
#             result = [
#                 (
#                     word2vec.key_to_index[token]
#                     if token in word2vec.key_to_index
#                     else max_token
#                 )
#             ]
#             children = node.children
#             for child in children:
#                 result.append(tree_to_index(child))
#             return result

#         def tree_to_token(node):
#             token = node.token
#             if type(token) is bytes:
#                 token = token.decode("utf-8")
#             result = [token]
#             children = node.children
#             for child in children:
#                 result.append(tree_to_token(child))
#             return result

#         def trans2seq(r):
#             blocks = []
#             func(r, blocks)
#             tree = []
#             for b in blocks:
#                 btree = tree_to_index(b)
#                 token_tree = tree_to_token(b)
#                 tree.append(btree)
#             return tree

#         return trans2seq(data)

#     def generate_block_seqs(self, data, name: str):
#         blocks_path = None
#         if name == "train":
#             blocks_path = "subtrees/" + args.input + "/train_block.pkl"
#         elif name == "test":
#             blocks_path = "subtrees/" + args.input + "/test_block.pkl"
#         elif name == "dev":
#             blocks_path = "subtrees/" + args.input + "/dev_block.pkl"

#         from prepare_data import get_blocks as func
#         from gensim.models.word2vec import Word2Vec

#         word2vec = Word2Vec.load(self.w2v_path).wv
#         max_token = word2vec.vectors.shape[0]

#         def tree_to_index(node):
#             token = node.token
#             if type(token) is bytes:
#                 token = token.decode("utf-8")
#             result = [
#                 (
#                     word2vec.key_to_index[token]
#                     if token in word2vec.key_to_index
#                     else max_token
#                 )
#             ]
#             children = node.children
#             for child in children:
#                 result.append(tree_to_index(child))

#             if len(result) == 0:
#                 print("Empty tree found in tree_to_index function. Node:", node)
#             return result

#         def tree_to_token(node):
#             token = node.token
#             if type(token) is bytes:
#                 token = token.decode("utf-8")
#             result = [token]
#             children = node.children
#             for child in children:
#                 result.append(tree_to_token(child))
#             if len(result) == 0:
#                 print("Empty tree found in tree_to_token function. Node:", node)
#             return result

#         def trans2seq(r):
#             # print("Input to trans2seq function:", r)
#             blocks = []
#             func(r, blocks)
#             tree = []
#             for b in blocks:
#                 btree = tree_to_index(b)
#                 token_tree = tree_to_token(b)
#                 tree.append(btree)
#             # if len(tree) == 0:
#             #     print("Empty tree found in data:", r)
#             return tree

#         trees = data
#         trees["code"] = trees["code"].apply(trans2seq)

#         empty_trees_count = trees["code"].apply(lambda x: len(x) == 0).sum()
#         print("Number of empty trees in", name, "data:", empty_trees_count)
#         print("Number of rows in", name, "data:", len(trees))
#         trees.to_pickle(blocks_path)
#         return trees

#     def run(self, dataset):
#         print("parse source code...")
#         self.parse_source(dataset)
#         print("train word2vec model...")
#         self.dictionary_and_embedding(size=128)
#         print("generate block sequences...")
#         self.train_block = self.generate_block_seqs(self.train_keep, "train")
#         self.dev_block = self.generate_block_seqs(self.dev_keep, "dev")
#         self.test_block = self.generate_block_seqs(self.test_keep, "test")


# if __name__ == "__main__":
#     ppl = Pipeline()
#     print("Now processing dataset: ", args.input)
#     ppl.run(args.input)

import argparse
import copy
import pandas as pd
import os

from tree_sitter import Language, Parser
from prepare_data import get_root_paths, get_sequences, get_blocks


def parse_options():
    parser = argparse.ArgumentParser(description="TrVD preprocess~.")
    parser.add_argument(
        "-i",
        "--input",
        default="mutrvd",
        choices=["mutrvd"],
        help="training dataset type",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    return args


def parse_ast_js(source):
    JS_LANGUAGE = Language("build_languages/my-languages.so", "javascript")
    parser = Parser()
    parser.set_language(JS_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    # if not tree or not tree.root_node or tree.root_node.type == "ERROR":
    #     print("Empty or error tree found for source:/n")
    return tree


args = parse_options()


class Pipeline:
    def __init__(self):
        self.train = None
        self.train_keep = None
        self.train_block = None
        self.dev = None
        self.dev_keep = None
        self.dev_block = None
        self.test = None
        self.test_keep = None
        self.test_block = None
        self.size = None
        self.w2v_path = None

    def parse_source(self, dataset):
        # train = pd.read_pickle("dataset/" + dataset + "/train.pkl")
        test = pd.read_pickle("dataset/" + dataset + "/test.pkl")
        # dev = pd.read_pickle("dataset/" + dataset + "/val.pkl")

        # print("Number of rows in train dataset:", len(train))
        print("Number of rows in test dataset:", len(test))
        # print("Number of rows in dev dataset:", len(dev))

        # train["code"] = train["code"].apply(parse_ast_js)
        # self.train = train
        # self.train_keep = copy.deepcopy(train)
        # dev["code"] = dev["code"].apply(parse_ast_js)
        # self.dev = dev
        # self.dev_keep = copy.deepcopy(dev)
        test["code"] = test["code"].apply(parse_ast_js)
        self.test = test
        self.test_keep = copy.deepcopy(test)

    def dictionary_and_embedding(self, size):
        self.size = size
        trees = self.train
        self.w2v_path = "subtrees/" + args.input + "/node_w2v_" + str(size)
        if not os.path.exists("subtrees"):
            os.mkdir("subtrees")
        if not os.path.exists("subtrees/" + args.input):
            os.mkdir("subtrees/" + args.input)

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)

            paths = []
            get_root_paths(ast, paths, [])

            paths.append(sequence)
            return paths

        if not os.path.exists(self.w2v_path):
            corpus = trees["code"].apply(trans_to_sequences)
            paths = []
            for all_paths in corpus:
                for path in all_paths:
                    path = [
                        token.decode("utf-8") if type(token) is bytes else token
                        for token in path
                    ]
                    paths.append(path)
            corpus = paths
            from gensim.models.word2vec import Word2Vec

            print("corpus size: ", len(corpus))
            w2v = Word2Vec(corpus, vector_size=size, workers=96, sg=1, min_count=3)
            print("word2vec : ", w2v)
            w2v.save(self.w2v_path)

    def generate_block_seqs(self, data, name: str):
        blocks_path = None
        if name == "train":
            blocks_path = "subtrees/" + args.input + "/train_block.pkl"
        elif name == "test":
            blocks_path = "subtrees/" + args.input + "/test_block.pkl"
        elif name == "dev":
            blocks_path = "subtrees/" + args.input + "/dev_block.pkl"

        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.w2v_path).wv
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode("utf-8")
            result = [
                (
                    word2vec.key_to_index[token]
                    if token in word2vec.key_to_index
                    else max_token
                )
            ]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))

            return result

        def tree_to_token(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode("utf-8")
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_token(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks(r, blocks)
            # if not blocks:
            #     print("No blocks extracted for AST:")
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                # if not btree:
                #     print("Empty btree found for block:", b)
                token_tree = tree_to_token(b)
                # if not token_tree:
                #     print("Empty token_tree found for block:", b)
                tree.append(btree)
            # if not tree:
            #     print("Empty tree found in data:", r)
            return tree

        trees = data
        trees["code"] = trees["code"].apply(trans2seq)

        empty_trees_count = trees["code"].apply(lambda x: len(x) == 0).sum()
        print("Number of empty trees in", name, "data:", empty_trees_count)
        print("Number of rows in", name, "data:", len(trees))
        trees.to_pickle(blocks_path)
        return trees

    def run(self, dataset):
        print("parse source code...")
        self.parse_source(dataset)
        print("train word2vec model...")
        self.dictionary_and_embedding(size=128)
        print("generate block sequences...")
        # self.train_block = self.generate_block_seqs(self.train_keep, "train")
        # self.dev_block = self.generate_block_seqs(self.dev_keep, "dev")
        self.test_block = self.generate_block_seqs(self.test_keep, "test")


if __name__ == "__main__":
    ppl = Pipeline()
    print("Now processing dataset: ", args.input)
    ppl.run(args.input)
