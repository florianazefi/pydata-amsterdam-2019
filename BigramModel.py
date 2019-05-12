# MIT License; see ./LICENSE

# AUTHOR : Floriana ZEFI
# CONTACT : florianagjzefi@gmail.com or floriana.zefi@ing.com
# PUBLICATION DATE : 12-05-2019

import time
import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class BigramModel(object):
    """
    Class to calculate transition probabilities based
    on the Markov chain approximation
    Atributes:
        fit:
        get_cond_bigram_proba:
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, sequences, seq_ids):
        """
        fit the sequences and ids

        Args:
            sequences: encoded states for each applicant
            seq_ids: the identification number for the each applicant
        Return:
            df: bigram counts based on their frequency
        """
        self.encoder = LabelEncoder()
        self.encoder.fit(sequences)
        self.n_states = len(self.encoder.classes_)

        self.n_train = len(sequences)

        sequences = self.encoder.transform(sequences)

        df = pd.DataFrame(data={"state": sequences, "id": seq_ids})
        df["bigram"] = list(zip(df["state"], np.roll(df["state"], -1)))

        df = df.groupby("id").apply(lambda x: x[:-1])

        # Count the states and bigrams
        bigram_counts = collections.Counter(df["bigram"])
        single_counts = collections.Counter(sequences)

        def get_cond_bigram_proba(i, j):
            """
            Calculates the conditional probability
            Args:
                (i, j) the i-th and j-th entry of the matrix
            Return:
                probability (float)
            """
            if not i in single_counts and not j in single_counts:
                return 1.0 / self.n_train / self.alpha
            if not j in single_counts:
                return 1.0 / single_counts[i] / self.alpha
            if not i in single_counts:
                return 1.0 / single_counts[j] / self.alpha
            if not (i, j) in bigram_counts:
                return 1.0 / single_counts[i] / self.alpha

            numerator = bigram_counts[(i, j)]
            denumerator = sum([counts for bigram, counts in bigram_counts.items() if bigram[0] == i])

            return float(numerator) / denumerator

        # Calculate probability matrix
        get_cond_bigram_proba_v = np.vectorize(get_cond_bigram_proba)

        # The row and column indices of the probability matrix. There is one
        # more row and column different states in the training dataset, which
        # will correspond to previously unseen values.
        x = range(self.n_states + 1)

        self.probas = get_cond_bigram_proba_v(*np.meshgrid(x, x))

        # Normalize
        self.probas /= np.sum(self.probas, axis=0)

        self.log_probas = np.log10(self.probas)

        # Create DataFrames with bigram and single counts
        self.bigram_counts_df = pd.DataFrame.from_dict(bigram_counts, orient="index").reset_index()
        self.bigram_counts_df.columns = ["bigram", "frequency"]
        self.bigram_counts_df = self.bigram_counts_df.sort_values("frequency", ascending=False)

        self.single_counts_df = pd.DataFrame.from_dict(single_counts, orient="index").reset_index()
        self.single_counts_df.columns = ["state", "frequency"]
        self.single_counts_df = self.single_counts_df.sort_values("frequency", ascending=False)

        # Do the inverse label transformation such that labels in the
        # DataFrames are like in the original data
        self.encoder.inverse_transform(self.single_counts_df["state"])
        self.bigram_counts_df["bigram"] = list(
            zip(*[self.encoder.inverse_transform(l) for l in zip(*self.bigram_counts_df["bigram"])])
        )

        return self

    def predict_grouped_probas(self, sequences, seq_ids, log=True):
        # classes that were seen in the training get transformed with the
        # encoder, unseen classes get replaced with n_states, the placeholder
        # for useen classes
        seen_mask = np.in1d(sequences, self.encoder.classes_)
        sequences[seen_mask] = self.encoder.transform(sequences[seen_mask])
        sequences[~seen_mask] = self.n_states

        df = pd.DataFrame(data={"state": sequences, "id": seq_ids})

        M = self.log_probas if log else self.probas
        df["prob"] = M[df["state"], np.roll(df["state"], -1)]

        df = df.groupby("id")["prob"].apply(lambda x: x[:-1])

        return df.groupby("id")

    def predict_mean_proba(self, sequences, seq_ids, log=True):
        """
        Calculate the mean probability of a sequence
        Args:
            sequences:
            seq_ids:
            log (bool): if True, returns the log probabilities
        Return:
            mean probability for a sequence
        """
        return self.predict_grouped_probas(sequences, seq_ids, log).apply(np.mean)

    def predict_proba_stats(self, sequences, seq_ids):
        """
        Args:
            sequences:
            seq_ids:
            log: (bool) if True it returns the log of the probability
        Return:
            state probabilities
        """
        probas = self.predict_grouped_probas(sequences, seq_ids, log=False)
        log_probas = self.predict_grouped_probas(sequences, seq_ids, log=True)

        df = pd.DataFrame(
            data={
                "V_Markov_seq_prob_log_tot": log_probas.apply(np.sum),
                "V_Markov_seq_prob_log_mean": np.log10(probas.apply(np.mean)),
                "V_Markov_seq_prob_log_median": np.log10(probas.apply(np.median)),
                "V_Markov_seq_prob_mean": probas.apply(np.mean),
                "V_Markov_seq_prob_median": probas.apply(np.median),
                "V_Markov_seq_prob_min": probas.apply(np.min),
                "V_Markov_seq_prob_max": probas.apply(np.max),
            }
        )
        df.index = probas.apply(np.mean).index
        return df.reset_index().rename(columns={"id": "person_id"})

    def plot_frequency(self, n_most_frequent):
        """
        Function to plot ngram and frequency
        Args:
            df: Pandas dataframe
        Return:
            plot (int): top n most frequent ngram vs frequency
        """
        self.bigram_counts_df.head(n_most_frequent)[::-1].plot.barh("bigram", "frequency", color="teal")
        plt.gca().get_legend().remove()

    def plot_proba_matrix(self, log=True, proba_unseen=False):
        """
        Function to plot probability and counts in 2-D plot
        Args:
            probas: probability matrix
         Return:
             2D plot
        """
        M =  self.log_probas if log else self.probas
        plt.imshow(M if proba_unseen else M[:-1,:-1], origin="lower")

        # Set axis labels to the actual state indices
        n = self.n_states
        n_labels = n
        step = int(n / (n_labels - 1))
        positions = np.arange(0, n + 1 * proba_unseen, step)
        labels = list(self.encoder.classes_[::step]) + ["new"] * proba_unseen
        plt.xticks(positions, labels)
        plt.yticks(positions, labels)

        colorbar = plt.colorbar()
        plt.ylabel(r"$w_i$")
        plt.xlabel(r"$w_{i-1}$")
        if log:
            colorbar.set_label(r"log $P(w_{i-1}|w_i)$")
        else:
            colorbar.set_label(r"$P(w_{i-1}|w_i)$")
