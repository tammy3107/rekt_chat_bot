import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class DutchTranslate(text_problems.Text2TextProblem):
    """ Predict the dutch translation from english sentence """

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):

        return [{"split": problem.DatasetSplit.TRAIN, "shards": 9,}, {"split": problem.DatasetSplit.EVAL, "shards": 1, }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        del data_dir
        del tmp_dir
        del dataset_split

        """
        with open("/home/manish/vishal/t2t_experiment/data/eng_sent.txt") as f:
            content_eng=f.readlines()
        
        with open("/home/manish/vishal/t2t_experiment/data/dutch_sent.txt") as f:
            content_dutch=f.readlines()

        for eng, dutch in zip(content_eng, content_dutch):
            eng=eng.lower()
            dutch=dutch.lower()
            yield{
                    "inputs": eng,
                    "targets": dutch,
                    }
        """
        with open("/content/jokes_final.txt") as f:
            content=f.readlines()
        for line in content:
            sent=line.split("\t")
            input_sent=str(sent[0]).rstrip()
            target=str(sent[1]).rstrip()
            yield{
                    "inputs": input_sent,
                    "targets": target,
                    }
