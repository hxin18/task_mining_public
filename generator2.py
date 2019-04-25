import json
import os
import re
import string
import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant

dict_ = enchant.Dict("en_US")
import networkx as nx
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import codecs

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class MergeGenerator:

	def __init__(self, input_json):
		self.id_to_text = {}
		self.commit_graph = {}
		self.id_to_module = {}
		self.id_to_direct_model = {}
		self.org_tree = {}
		self.id_to_rank = {}
		self.id_to_ngram = {}
		self.vectorizer = TfidfVectorizer()
		self.vectorized = {}
		self.id_to_time = {}
		self.issue_to_id = {}
		self.id_to_issue = {}
		with codecs.open(input_json, "r") as f:
			for line in f:
				commit = json.loads(line)
				ha = commit["hash"]
				self.id_to_text[ha] = commit["text"]
				self.id_to_module[ha] = commit["module"]
				self.id_to_time[ha] = commit["time"]
				gc_issue = commit["issue"]
				if len(gc_issue) == 1:
					issue_no = gc_issue[0]
					if issue_no not in self.issue_to_id:
						self.issue_to_id[issue_no] = []
					self.issue_to_id[issue_no].append(ha)
					self.id_to_issue[ha] = issue_no
				parent = commit["parent"]
				child = commit["child"]
				for c in child:
					if c not in self.commit_graph:
						self.commit_graph[c] = set()
					self.commit_graph[c].add(ha)
				for p in parent:
					if ha not in self.commit_graph:
						self.commit_graph[ha] = set()
					self.commit_graph[ha].add(p)

		self.org_tree["."] = set()
		for id_ in self.id_to_module:
			module = self.id_to_module[id_]
			parent_set = set(".")
			child_set = set()
			for p in module:
				if p == ".":
					continue
				p_list = p.split("/")
				if (len(p_list) == 1):
					self.org_tree["."].add(p_list[-1])
					child_set.add(p_list[-1])
				for i in range(1, len(p_list)):
					parent = "/".join(p_list[:i])
					child = "/".join(p_list[:i + 1])
					parent_set.add(parent)
					child_set.add(child)
					if parent not in self.org_tree:
						self.org_tree[parent] = set()
					self.org_tree[parent].add(child)
			for m in child_set:
				if m not in parent_set:
					if m not in self.org_tree:
						self.org_tree[m] = set()
					self.org_tree[m].add(id_)
					if id_ not in self.id_to_direct_model:
						self.id_to_direct_model[id_] = set()
					self.id_to_direct_model[id_].add(m)

	@staticmethod
	def ngram_word(word):
		list_res = []
		word_ = "__" + word + "__"
		for i in range(0, len(word_) - 2):
			list_res.append(word_[i:i + 3])
		return list_res

	def generate_ngram(self, word):
		list_res = []
		punctuation = set(string.punctuation)
		temp = []
		for letter in word:
			if not letter in punctuation:
				temp.append(letter)
		word = ''.join(temp)
		word_list = word.split()
		for words in word_list:
			l = self.ngram_word(words)
			for ii in l:
				list_res.append(''.join(filter(lambda x: ord(x) < 128, ii)))
		return str(" ".join(list_res))

	def generate_word_feature(self):
		for i in self.id_to_text:
			self.id_to_ngram[i] = self.generate_ngram(self.id_to_text[i])
		vectorize = self.vectorizer.fit(list(self.id_to_ngram.values()))
		for i in self.id_to_text:
			self.vectorized[i] = vectorize.transform([self.id_to_ngram[i]])


	def run(self):
		self.generate_word_feature()
		G = nx.Graph()
		id_to_node = {}
		node_to_id = {}
		idx_ = 1
		for i in self.id_to_time:
			G.add_node(idx_, name=i)
			id_to_node[i] = idx_
			node_to_id[idx_] = i
			idx_+=1

		two_hops = {}
		for c in self.commit_graph:
			two_hops[c] = set()
			for ch in self.commit_graph[c]:
				two_hops[c].add(ch)
				if ch in self.commit_graph:
					for chh in self.commit_graph[ch]:
						two_hops[c].add(chh)
		idxx = 0
		for i in self.id_to_time:
			idxx+=1
			if idxx%200==0:
				print(idxx)
			if i in two_hops:
				for j in two_hops[i]:
					if i in self.id_to_issue and j in self.id_to_issue and self.id_to_issue[i] != self.id_to_issue[j]:
						continue
					if cosine_similarity(self.vectorized[i],self.vectorized[j])[0][0]>0.55:
						G.add_edge(id_to_node[i], id_to_node[j])
		# for issue in self.issue_to_id:
		# 	issue_list = self.issue_to_id[issue]
		# 	for idx, iss in enumerate(issue_list):
		# 		if idx == len(issue_list)-1:
		# 			break
		# 		G.add_edge(id_to_node[issue_list[idx]], id_to_node[issue_list[idx+1]])

		with open("apache_http_no.json", "w") as f:
			for i in nx.connected_components(G):
				if (len(i) > 1):
					cluster = {}
					cluster["commits"] = []
					cluster["detail"] = {}
					cluster["issue"] = {}
					cluster["time"] = {}
					for c in i:
						cluster["commits"].append(node_to_id[c])
						cluster["detail"][node_to_id[c]] = self.id_to_text[node_to_id[c]]
						cluster["time"][node_to_id[c]] = self.id_to_time[node_to_id[c]]
						if node_to_id[c] in self.id_to_issue:
							cluster["issue"][node_to_id[c]] = self.id_to_issue[node_to_id[c]]
					f.write(json.dumps(cluster, indent=4) + "\n");

aa = MergeGenerator("/Users/xinhuang/Documents/dr/data_transformer/apache_http_clean.json")
aa.run()