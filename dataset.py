import glob, os, re
#from tqdm import tqdm
from cleantext import clean
import ftfy
import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Manager, Pool
import tensorflow_text as tft
from shutil import copyfile
import tensorflow_text as tft
import json
from sacrerouge.metrics import Rouge
from rouge_score import rouge_scorer
from nltk import tokenize
from collections import Counter
from rouge_metric import PyRouge

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


def compute_stats(data_in, id):
    return [
            data_in[:, id].mean(),
            np.quantile(data_in[:, id], 0.5),
            np.quantile(data_in[:, id], 0.9),
            np.quantile(data_in[:, id], 0.1),
            np.max(data_in[:, id]),
            np.min(data_in[:, id])
            ]



class Dataset():
    def __init__(self ,
                 output_dir='./',
                 root_dir = "./",
                 process_n=10, clean_deep=True):
        self.output_dir = output_dir
        self.root_dir = root_dir
        self._create_dir(output_dir)
        self.process_n = process_n
        self.clean_deep = clean_deep

    def _create_dir(self, output_dir):
        '''check or create new dir'''
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

    def _list_file(self, data_dir):
        '''list all txt files in data_dir'''
        data_dir = os.path.join(self.root_dir, data_dir)
        return list(glob.glob(
                            os.path.join(data_dir, "*.txt")
                              ))

    def _text_to_file(self, text, out_file):
        ''' save text to a file'''
        f = open(out_file, "w", encoding="utf-8" )
        f.write(text)
        f.close()

    def _out_file_path(self, out_dir, orig_name):
        '''create new output path from old path of file'''
        return os.path.join(self.output_dir, out_dir, os.path.basename(orig_name))

    def sentence_check(self, sentence):
        n_upper = sum(map(str.isupper, sentence))
        n_lower = sum(map(str.islower, sentence))
        perc_up = n_upper / (n_upper + n_lower + 1)
        alphanum = len("".join(filter(str.isalnum, sentence)))
        prc_alphanum = alphanum / ((len(sentence)) + 1)
        return perc_up < 0.5 and prc_alphanum > 0.5

    def clean_dir(self, data_dir, cleaning=True):
        '''
        clean all files in one directory
        '''
        data_dir = os.path.join(self.root_dir, data_dir)
        self.dir_prefix_tmp = os.path.basename(os.path.normpath(data_dir))
        dir_out = os.path.join(self.output_dir, self.dir_prefix_tmp)
        self._create_dir(dir_out)
        # read every file and clean it
        files = self._list_file(data_dir)
        #skip fils done already
        files_done = [os.path.basename(f) for f in self._list_file(dir_out)]
        files = [f for f in files if os.path.basename(f) not in files_done]
        #manager = Manager()
        if not files:
            return
        #self.text_dict = manager.dict()
        p = Pool(self.process_n)
        p.map(self.clean_dir_fn, files)

    def clean_dir_fn(self, f_name, cleaning=True):
        '''
        clen all files in data_dir and save them in output_dir
        '''

        #dir_prefix = os.path.basename(os.path.normpath(data_dir))
        #self._create_dir(os.path.join(self.output_dir, dir_prefix))
        #read every file and clean it
        #files = self._list_file(data_dir)
        #for f_name in tqdm.tqdm(files):
        with open(f_name, "r", encoding="utf-8") as f_in:
            raw_text = f_in.read()
            clean_text = self.clean_file(raw_text)
            self._text_to_file(clean_text,
                               self._out_file_path(self.dir_prefix_tmp, f_name)
                               )

        logging.info('saved file %s' % (f_name))

    def count_uniq_doc(self, data_dir):
        data_dir = os.path.join(self.root_dir, data_dir)
        files = self._list_file(data_dir)
        files_dict = {}
        for f_name in files:
            f_rel_name = os.path.splitext(os.path.basename(f_name))[0]
            file_n, sec_n, topic = f_rel_name.split('_')
            files_dict[file_n] = 1
        tot_doc = len(files_dict)
        logging.info('total number of unique documents: %d' % (tot_doc))
        return tot_doc

    def check_missing_summ(self, sect_dir, data_dir, narrative_only=True):
        '''check files in section dir not yet done in summary dir'''
        data_dir = os.path.join(self.root_dir, data_dir)
        sect_dir = os.path.join(self.root_dir, sect_dir)
        files = self._list_file(data_dir)
        files_dict = {}
        for f_name in files:
            f_rel_name = os.path.splitext(os.path.basename(f_name))[0]
            file_n, sec_n, topic = f_rel_name.split('_')
            files_dict[file_n] = 1

        sect_files = self._list_file(sect_dir)
        sect_dict = {}
        for f_name in sect_files:
            f_rel_name = os.path.splitext(os.path.basename(f_name))[0]
            file_n, sec_n, topic = f_rel_name.split('_')
            #if (not narrative_only) or (narrative_only and int(topic)>0):#useless
            sect_dict[file_n] = 1
        doc_diff = set(sect_dict.keys())-set(files_dict.keys())
        print(doc_diff)
        logging.info('total number of unique documents to do: %d' % (len(doc_diff)))
        return list(doc_diff)

    def doc_from_section_fn(self, files):
        '''function called b every thread/process to merge section for 1 doc'''
        sec_n_tmp, file_n_tmp = 0, 0
        text = ""
        for f_name in files:
            f_rel_name = os.path.splitext(os.path.basename(f_name))[0]
            file_n, sec_n, topic = f_rel_name.split('_')
            if int(sec_n)<=sec_n_tmp:
                logging.info('error file %s, %d <%d' % (f_name, int(sec_n),sec_n_tmp))
            assert(sec_n_tmp < int(sec_n))
            assert((file_n_tmp == 0) | (file_n_tmp == int(file_n)))

            sec_n_tmp = int(sec_n)
            file_n_tmp = int(file_n)
            if (not self.narrative_only) or (self.narrative_only and int(topic) > 0):
                raw_text = open(f_name, "r", encoding="utf-8").read()
                text = text + " " + raw_text
        if self.clean_text:
            clean_text = self.clean_file(text)
        self._text_to_file(clean_text,
                           self._out_file_path(self.dir_prefix_tmp + self.post_fix_merge, str(file_n) + '.txt')
                           )
        logging.info('saved file %s' % file_n )

    def _doc_files_array(self, data_dir, narrative_only=False):
        '''returns an array of list of sections file
           Every row are the sections from one document
        '''
        data_dir = os.path.join(self.root_dir, data_dir)
        files = self._list_file(data_dir)
        #files = sorted(files)
        topic_n_from_name = lambda f_name: int(os.path.splitext(os.path.basename(f_name))[0].split('_')[2])
        if narrative_only:
            files = [f for f in files if topic_n_from_name(f)>0]
        doc_n_from_name = lambda f_name: int(os.path.splitext(os.path.basename(f_name))[0].split('_')[0])
        doc_n = [doc_n_from_name(f_name) for f_name in files]
        #better to sort on integers
        int_sorted = np.argsort(doc_n)
        doc_n = [doc_n[n] for n in int_sorted]
        files = [files[n] for n in int_sorted]
        doc_c_unq, idx = np.unique(doc_n, return_index=True)
        idx_sorted = np.argsort(idx)
        idx = idx[idx_sorted]
        doc_c_unq = doc_c_unq[idx_sorted]
        files_docid = [files[i:j] for i, j in zip(idx, np.concatenate((idx[1:], np.array([len(files)]))))]
        files_docid = [sorted(f) for f in files_docid]  ##in case are not saved in order
        return doc_c_unq, files_docid

    def doc_from_sections(self, data_dir, narrative_only=True, cleaning=True, clean_deep=True):
        '''
        reads section files in a directory and merge them skipping non narrative sections
        '''
        data_dir = os.path.join(self.root_dir, data_dir)
        self.clean_deep = clean_deep
        self.dir_prefix_tmp = os.path.normpath(data_dir)
        self.post_fix_merge = '_merge'
        self._create_dir(os.path.join(self.output_dir, self.dir_prefix_tmp + self.post_fix_merge))

        _, files_docid = self._doc_files_array(data_dir)
        self.narrative_only = narrative_only
        self.clean_text = cleaning
        #manager = Manager()
        # self.text_dict = manager.dict()
        n_thread = 12
        p = Pool(n_thread)
        p.map(self.doc_from_section_fn, files_docid)


    def clean_file(self, text):
        '''clean file with ad-hoc rules'''
        #clean page numbers
        text = " ".join([line for line in text.splitlines() if not line.strip().isdigit()])
        # clean new lines
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        # clean sequence of 3+ symbols, substitute with one only
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        if self.clean_deep:
            #clean sentence
            #text = text.encode("ascii", "ignore").encode("ascii", "ignore")
            text = ftfy.fix_encoding(text)

            #clean pattern - number followed by letter, letter followed by number
            text = re.sub(r'([0-9$%€£)]{1})([A-Za-z]{1})', r'\1 \2', text)
            text = re.sub(r'([A-Za-z]{1})([(0-9$%€£]{1})', r'\1 \2', text)
            #clean from consecutive 4+ all upper cases words and numbers,
            text = re.sub(r'((\b[A-Z0-9]{2,}\b ){4,})', r'', text)
            #clean from consecutive 5 numbers. no!!!
            #text = re.sub(r'((\b[0-9]{1,}\b ){5,})', r'', text)
            #consecutive not alpha numeric symbols
            text = re.sub('([^0-9a-zA-Z]+ ){3,}', '', text)
        #split in sentences
        sentences = tokenize.sent_tokenize(text)
        if self.clean_deep:
            sentences = [s for s in sentences if self.sentence_check(s)]
        if self.clean_deep:
            s_count = Counter(sentences)
            sentences = [s for s in sentences if s_count[s]==1]
        text = " ".join(sentences)
        text = re.sub(' +', ' ', text)
        if self.clean_deep:
            #clean different things
            text = clean(text,
                         fix_unicode=True,  # fix various unicode errors
                         to_ascii=True,  # transliterate to closest ASCII representation
                         lower=True,  # lowercase text
                         no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                         no_urls=False,  # replace all URLs with a special token
                         no_emails=True,  # replace all email addresses with a special token
                         no_phone_numbers=True,  # replace all phone numbers with a special token
                         no_numbers=False,  # replace all numbers with a special token
                         no_digits=False,  # replace all digits with a special token
                         no_currency_symbols=False,  # replace all currency symbols with a special token
                         no_punct=False,  # remove punctuations
                         replace_with_punct="",  # instead of removing punctuations you may replace them
                         replace_with_url="",
                         replace_with_email="",
                         replace_with_phone_number="",
                         replace_with_number="<NUMBER>",
                         replace_with_digit="0",
                         replace_with_currency_symbol="<CUR>",
                         lang="en"  # set to 'de' for German special handling
                         )

        return text

    def compare_len_fn(self, f_name):

        with open(f_name[0], "r", encoding="utf-8") as f_in1, open(f_name[1], "r", encoding="utf-8") as f_in2:
            text1 = f_in1.read()
            text2 = f_in2.read()
            self.lengths[f_name[0]] = [len(text1.split()),
                                       len(text1.split(".")),
                                       len(text2.split()),
                                       len(text2.split("."))
                                      ]
        logging.info('loaded %s' % (f_name[0]))

    def compare_len(self, dir1, dir2):
        '''compare full txt vs narrative only sections'''
        dir1 = os.path.join(self.root_dir, dir1)
        dir2 = os.path.join(self.root_dir, dir2)
        files = self._list_file(dir1)
        lengths = {}
        manager = Manager()
        self.lengths = manager.dict()
        p = Pool(self.process_n)
        #get the name of the file for full text
        files2 = [os.path.splitext(os.path.basename(f_name))[0] for f_name in files]
        files2 = [os.path.join(dir2, f_name+'.txt') for f_name in files2]
        files = zip(files, files2)
        p.map(self.compare_len_fn, files)

        lengths_array = np.asarray(list(self.lengths.values()))
        # compute_stats = lambda data_in, id : [
        #                                     data_in[:, id].mean(),
        #                                     np.quantile(data_in[:, id], 0.5),
        #                                     np.quantile(data_in[:, id], 0.9),
        #                                     np.quantile(data_in[:, id], 0.1),
        #                                     np.max(data_in[:, id]),
        #                                     np.min(data_in[:, id])
        #                                      ]
        for n, txt in enumerate(['Original text', 'After cutting non narrative sections']):
            for idx in [0, 1]:

                logging.info('{}: avg length {}, '\
                             'median length {}, 90th quantile {}, '\
                             '10th quantile, {}, max:{}, min: {}'.format(txt, *compute_stats(lengths_array, (2*n)+idx)
                                                                         )
                            )
                txt = txt + 'sentences'


        checked = lengths_array[:,0]>0
        logging.info('perc of narrative only text: avg %f' % ((lengths_array[checked,2]/lengths_array[checked,0]).mean()))

    def dir_stats_fn(self, f_name):
        '''dir stats jobs'''
        with open(f_name, "r", encoding="utf-8") as f_in1:
            text1 = f_in1.read()
            self.lengths[f_name] = [len(text1.split()),
                                       len(text1.split(".")),
                                      ]
        logging.info('loaded %s' % (f_name))



    def dir_stats(self, dir1):
        '''dir stats'''
        dir1 = os.path.join(self.root_dir, dir1)
        files = self._list_file(dir1)
        lengths = {}
        manager = Manager()
        self.lengths = manager.dict()
        p = Pool(self.process_n)
        p.map(self.dir_stats_fn, files)

        lengths_array = np.asarray(list(self.lengths.values()))

        for n, txt in enumerate(['file stats']):
            for idx in [0, 1]:

                logging.info('{}: avg length {}, '\
                             'median length {}, 90th quantile {}, '\
                             '10th quantile, {}, max:{}, min: {}'.format(txt, *compute_stats(lengths_array, (2*n)+idx)
                                                                         )
                            )
                txt = txt + 'sentences'

    def set_vocab_file(self, vocab_model_file):
        '''set vocabulary file'''
        self.vocab_model_file = vocab_model_file

    def filter_narrative_only(self, files):
        return [f for f in files if f[-5]!='0']

    def tokenize_dir_stats(self, data_dir, vocab_model_file, narrative_only=False):
        '''tokenize dir stats'''
        data_dir = os.path.join(self.root_dir, data_dir)
        files = self._list_file(data_dir)
        if narrative_only:
            files = self.filter_narrative_only(files)
        manager = Manager()
        self.token_stats = manager.dict()
        self.set_vocab_file(vocab_model_file)
        n_process = 5  #max 5 or it chrashes
        p = Pool(n_process)
        #self.tokenize_file_fn(files[0])
        p.map(self.tokenize_file_fn, files)
        lengths_array = np.asarray(list(self.token_stats.values()))

        logging.info('files words: avg length {}, ' \
                     'median length {}, 90th quantile {}, ' \
                     '10th quantile, {}, max:{}, min: {}'.format( *compute_stats(lengths_array, 1)
                                                                 )
                     )
        logging.info('files tokens: avg length {}, ' \
                     'median length {}, 90th quantile {}, ' \
                     '10th quantile, {}, max:{}, min: {}'.format(*compute_stats(lengths_array, 0)
                                                                 )
                     )
        checked = lengths_array[:, 1] > 0

        logging.info(
            'ratio n_tokens/n_words %f' % ((lengths_array[checked, 0] / lengths_array[checked, 1]).mean()))



    def tokenize_file_fn(self, file_name):
        '''tokenize function for a single process'''
        text = open(file_name, "r", encoding="utf-8").read()
        ids = self.tokenize_text(text)
        words_n = len(text.split())
        self.token_stats[file_name] = [ids.shape.as_list()[0], words_n]

    def tokenize_file(self, file_name):
        '''tokenize a file'''
        text = open(file_name, "r", encoding="utf-8").read()
        ids = self.tokenize_text(text)
        return ids

    def tokenize_text(self, text):
        '''tokenize a text'''
        tokenizer = tft.SentencepieceTokenizer(
            model=tf.io.gfile.GFile(self.vocab_model_file, "rb").read())
        #if substitute_newline:
        #    text = tf.strings.regex_replace(text, "\n", substitute_newline)
        ids = tokenizer.tokenize(text)
        logging.info('text words:%d , tokens: %d' % (len(text.split()), ids.shape.as_list()[0]))
        return ids

    def check_summ(self, goldsumm_dir, fulltxt_dir):
        goldsumm_dir = os.path.join(self.root_dir, goldsumm_dir)
        fulltxt_dir = os.path.join(self.root_dir, fulltxt_dir)
        '''try to check summary:
           1)check % of sentences in the summary which are copied in the original text
           2)check if the summary is done sentence by sentence or should by approach at a global level
        '''
        files = self._list_file(fulltxt_dir)

        files_summ = self._list_file(goldsumm_dir)
        files_summ = [os.path.basename(f) for f in files_summ]
        files = [os.path.basename(f) for f in files]
        lengths = []

    def _doc_score(self, scorer, target, pred):
        '''compute score for one sentence target and summary'''
        score = scorer.score(target.lower(), pred.lower())
        return (score['rouge1'].recall + score['rouge1'].fmeasure +
                2*score['rouge2'].recall + 2*score['rouge2'].fmeasure +
                1 * score['rouge3'].recall + 1 * score['rouge3'].fmeasure +
                2*score['rougeL'].recall + 2*score['rougeL'].fmeasure)/12

    def _doc_score_quick(self, scorer, target, pred):
        '''compute score for one sentence target and summary'''
        score = scorer.score(target.lower(), pred.lower())
        return (score['rouge1'].recall + score['rouge1'].fmeasure +
                2*score['rouge2'].recall + 2*score['rouge2'].fmeasure +
                3*score['rouge3'].recall + 3*score['rouge3'].fmeasure)/12

    def select_best_summary(self, summ_dir, fulltext_dir, quick=True):
        summ_dir = os.path.join(self.root_dir, summ_dir)
        fulltext_dir = os.path.join(self.root_dir, fulltext_dir)
        '''find best matching summary and save it to best_summary dir'''
        files_summ, summ_array = self._doc_files_array(summ_dir)
        best_dir = os.path.join(summ_dir, 'best')
        self._create_dir(best_dir)
        self._create_dir(os.path.join(summ_dir, 'score'))
        self.quick=quick
        best_done = self._list_file(best_dir)
        #filter out docs already done in best dir
        best_done = [int(os.path.basename(b).split("_")[0]) for b in best_done]
        summ_array = [s[1] for s in zip(files_summ,summ_array) if s[0] not in best_done]
        files_summ = [f for f in files_summ if f not in best_done]
        logging.info("still to do %d ..." % (len(files_summ)))
        manager = Manager()
        self.bestsummary_dict = manager.dict()
        self.fulltext_dir = fulltext_dir
        p = Pool(10)
        p.map(self._best_summary_fn, zip(files_summ, summ_array))

    def _best_summary_fn(self, files_array):
        if self.quick:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
        else:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
        fulltext_fname = os.path.join(self.fulltext_dir, str(files_array[0])+".txt")
        fulltext_file = open(fulltext_fname, "r", encoding="utf-8").read()
        summaries = [open(f, "r", encoding="utf-8").read() for f in files_array[1]]
        if self.quick:
            scores = [self._doc_score_quick(scorer, fulltext_file, s) for s in summaries]
        else:
            scores = [self._doc_score(scorer, fulltext_file, s) for s in summaries]
        n_best = np.argmax(scores)
        if isinstance(n_best, list):
            n_best = n_best[0]
        best_summary_fname = files_array[1][n_best]
        base_fname = os.path.basename(best_summary_fname)
        dst_fname = os.path.join(os.path.dirname(best_summary_fname), 'best')
        score_fname = os.path.join(os.path.dirname(best_summary_fname), 'score')
        dst_fname = os.path.join(dst_fname, base_fname)
        score_fname = os.path.join(score_fname, base_fname)
        copyfile(best_summary_fname, dst_fname)
        self._text_to_file("%.4f" % (scores[n_best]), score_fname)
        del scorer
        del fulltext_file
        del summaries

    def filter_best_summaries(self, bestsumm_dir):
        bestsumm_dir = os.path.join(self.root_dir, bestsumm_dir)
        #files_summ = [os.path.basename(f) for f in self._list_file(bestsumm_dir)]
        dir_score = os.path.join(os.path.dirname(bestsumm_dir), "score")
        score_files = self._list_file(dir_score)
        scores = [float(open(f, "r", encoding="utf-8").read()) for f in score_files]
        sort_idx = np.argsort(scores)
        low_q = int(0.05 * len(sort_idx))
        doc_ids = [os.path.basename(f).split("_")[0] for f in score_files]
        return [doc_ids[n] for n in sort_idx[low_q:]]

    def sec_to_summary(self, sect_dir, bestsumm_dir, version = 3, narrative_only=True, quick=True, keep_bestscores_summ_only=True):
        '''
        this function is matching every section to its sub-part in the summary
        some sections will not be summarised
        '''
        sect_dir = os.path.join(self.root_dir, sect_dir)
        bestsumm_dir = os.path.join(self.root_dir, bestsumm_dir)
        self.version = version
        self.quick = quick
        self.dir_prefix_tmp = os.path.basename(os.path.normpath(sect_dir))
        summary_dir = os.path.join(self.output_dir, self.dir_prefix_tmp+'_summaries'+str(self.version))
        summary_score_dir = os.path.join(self.output_dir, self.dir_prefix_tmp + '_summaries_scores'+str(self.version))
        self._create_dir(summary_dir)
        self._create_dir(summary_score_dir)
        files_docid, sec_array = self._doc_files_array(sect_dir, narrative_only)
        files_summ, summ_array = self._doc_files_array(bestsumm_dir)


        #sections dir and summary should contain same unique documents ids
        assert((files_docid==files_summ).all())
        files_todo = self.check_missing_summ(sect_dir, summary_dir, narrative_only)
        if keep_bestscores_summ_only:
            doc_ids_tokeep = self.filter_best_summaries(bestsumm_dir)
            files_todo = [f for f in files_todo if f in doc_ids_tokeep]
        if files_todo:
            sec_array = [s[1] for s in zip(files_docid, sec_array) if str(s[0]) in files_todo]
            summ_array = [s[1] for s in zip(files_summ, summ_array) if str(s[0]) in files_todo]
        #do not consider not narrative


        assert(len(sec_array) == len(summ_array))
        p = Pool(10)
        p.map(self._sec_to_summary_fn2, zip(sec_array, summ_array))
        #self._sec_to_summary_fn2(zip(sec_array, summ_array)[0])


    def _sentence_score(self, scorer, target, pred):
        '''compute score for one sentence target and summary'''
        score = scorer.score(target.lower(), pred.lower())
        if self.quick:
            score = (score['rouge1'].recall + score['rouge1'].fmeasure +
                2*score['rouge2'].recall + 2*score['rouge2'].fmeasure)
        else:
            score = (score['rouge1'].recall + score['rouge1'].fmeasure +
                2*score['rouge2'].recall + 2*score['rouge2'].fmeasure +
                2*score['rougeL'].recall + 2*score['rougeL'].fmeasure)/10
        return score

    def _sec_to_summary_fn2(self, sec_summ_files):
        '''
        function to match every summary sentence to its section
        Input: both sections and summary files
        '''

        summaries = [open(f, "r", encoding="utf-8").read() for f in sec_summ_files[1]]
        sections = [open(s, "r", encoding="utf-8").read() for s in sec_summ_files[0]]
        if self.quick:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        else:
            scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge2'], use_stemmer=True)
            scorer1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        #1 find summary best matching section
        #find sentences inside summary which are related to section
        sent_min_l = 30
        sent_max_l = 60
        def merge_sentences(sentences_list):
            sentences_merge = []
            buff = []
            for s in sentences_list:
                if len(buff)>sent_min_l:
                    if len(buff)<sent_max_l:
                        sentences_merge.append(" ".join(buff))
                        buff = []
                    else:
                        sentences_merge.append(" ".join(buff[:sent_max_l]))
                        buff = buff[sent_max_l:] #if it is 3*max_l fine
                else:
                    buff = buff + s.split(" ")
            sentences_merge.append(" ".join(buff))
            return sentences_merge


        ####cut sntences too long or merge sentences too short
        summ_sent_array = [merge_sentences(tokenize.sent_tokenize(s.replace('\n',''))) for s in summaries]
        sect_sent_array = [merge_sentences(tokenize.sent_tokenize(s.replace('\n',''))) for s in sections]
        del summaries
        del sections
        #1 for every sentence in summary identify the best matching sentence
        scores = []
        for n_summ, summ in enumerate(summ_sent_array):
            for n_summ_sent, sent_summ in enumerate(summ):
                #logging.info('processed summary n: %d, sentence n: %d'  % (n_summ, n_summ_sent))
                for n_sect, sect in enumerate(sect_sent_array):
                    for n_sect_sent, sect_sent in enumerate(sect):
                        score = self._sentence_score4(scorer, scorer1, sent_summ, sect_sent)
                        scores.append(pd.DataFrame(data=[[n_summ, n_summ_sent, n_sect,
                                                          n_sect_sent, score
                                                          ]],
                                                   columns=['n_summ', 'n_summ_sent',
                                                            'n_sect', 'n_sect_sent', 'score'
                                                            ]
                                                   ))
        #find max_match for every sentence in summary and tag as matched
        scores = pd.concat(scores, axis=0)
        scores.reset_index(inplace=True)
        scores['matched'] = False
        scores['matched_summ'] = ""
        gby_summ_sent = scores.groupby(['n_summ', 'n_summ_sent'])
        for key in gby_summ_sent.groups:
            grp = gby_summ_sent.get_group(key)
            max_match = grp['score'].idxmax()
            scores.loc[max_match, 'matched'] = True
            #store the matching sentence in the summary
            scores.loc[max_match, 'matched_summ'] = scores.loc[max_match, 'matched_summ'] + summ_sent_array[key[0]][key[1]]
        #for very section build summary summary
        # last two in sorting are only valid in case of single summary
        scores = scores.sort_values(['n_sect', 'n_sect_sent', 'n_summ', 'n_summ_sent'])
        gby_sect_sent = scores[scores['matched']].groupby(['n_sect'])
        for key in gby_sect_sent.groups:
            grp = gby_sect_sent.get_group(key)
            scores = grp['score'].values
            scores_ok = scores>0.0001
            sect_summ = ' '.join(grp['matched_summ'].values[scores_ok])
            if not scores_ok.any():
                continue
            sect_avg_score = np.mean(scores[scores_ok] )
            self._text_to_file(sect_summ,
                               self._out_file_path(os.path.join(self.output_dir,
                                                                self.dir_prefix_tmp+'_summaries'+str(self.version)),
                                                   sec_summ_files[0][key]
                                                   )
                               )
            self._text_to_file(str(sect_avg_score),
                               self._out_file_path(os.path.join(self.output_dir,
                                                                self.dir_prefix_tmp + '_summaries_scores'+str(self.version)),
                                                   sec_summ_files[0][key]
                                                   )
                               )
        logging.info('processed %s' % sec_summ_files[0][0])
        return True
    ###########################################################################
    ###########################################################################

    def sec_to_summary4(self, sect_dir, bestsumm_dir, version = 4, narrative_only=True, quick=False):
        '''
        this function is matching every section to its sub-part in the summary
        some sections will not be summarised
        '''
        sect_dir = os.path.join(self.root_dir, sect_dir)
        bestsumm_dir = os.path.join(self.root_dir, bestsumm_dir)
        self.version = version
        self.quick = quick
        self.dir_prefix_tmp = os.path.basename(os.path.normpath(sect_dir))
        summary_dir = os.path.join(self.output_dir, self.dir_prefix_tmp+'_summaries'+str(self.version))
        summary_score_dir = os.path.join(self.output_dir, self.dir_prefix_tmp + '_summaries_scores'+str(self.version))
        best_summary_dir = os.path.join(self.output_dir, self.dir_prefix_tmp + '_best_L_summaries' + str(self.version))
        self._create_dir(summary_dir)
        self._create_dir(summary_score_dir)
        self._create_dir(best_summary_dir)
        files_docid, sec_array = self._doc_files_array(sect_dir, narrative_only)
        files_summ, summ_array = self._doc_files_array(bestsumm_dir)


        #sections dir and summary should contain same unique documents ids
        assert((files_docid==files_summ).all())
        files_todo = self.check_missing_summ(sect_dir, summary_dir, narrative_only)
        # if keep_bestscores_summ_only:
        #     doc_ids_tokeep = self.filter_best_summaries(bestsumm_dir)
        #     files_todo = [f for f in files_todo if f in doc_ids_tokeep]
        if files_todo:
            sec_array = [s[1] for s in zip(files_docid, sec_array) if str(s[0]) in files_todo]
            summ_array = [s[1] for s in zip(files_summ, summ_array) if str(s[0]) in files_todo]
        #do not consider not narrative


        assert(len(sec_array) == len(summ_array))
        p = Pool(12)
        p.map(self._sec_to_summary_fn4, zip(sec_array, summ_array))

    def _sentence_score4(self, scorer, scorer1, target, pred):
        '''compute score for one sentence target and summary'''
        score = scorer.score(target.lower(), pred.lower())
        score = 4*score['rougeL'].recall*len(target) + score['rouge2'].recall*len(target)
        target_numeric = " ".join(re.findall(r'\d+', target))
        score1 = 6*scorer1.score(target_numeric, " ".join(re.findall(r'\d+', pred)))['rouge1'].recall*len(target_numeric)/(len(target)+1)

        return score+score1

    def _sec_to_summary_fn4(self, sec_summ_files):
        '''
        function to match every summary sentence to its section
        Input: both sections and summary files
        '''

        summaries = [open(f, "r", encoding="utf-8").read() for f in sec_summ_files[1]]
        sections = [open(s, "r", encoding="utf-8").read() for s in sec_summ_files[0]]
        if self.quick:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        else:
            scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge2'], use_stemmer=True)
            scorer1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        #1 find summary best matching section
        #find sentences inside summary which are related to section
        sent_min_l = 40
        sent_max_l = 120
        def merge_sentences(sentences_list):
            sentences_merge = []
            buff = []
            for s in sentences_list:
                #dop not include some sentences
                if self.sentence_check(s):
                    buff = buff + s.split(" ")
                if len(buff)>sent_min_l:
                    if len(buff)<sent_max_l:
                        sentences_merge.append(" ".join(buff))
                        buff = []
                    else:
                        sentences_merge.append(" ".join(buff[:sent_max_l]))
                        buff = buff[sent_max_l:] #if it is 3*max_l fine

            sentences_merge.append(" ".join(buff))
            return sentences_merge


        ####cut sntences too long or merge sentences too short
        summ_sent_array = [merge_sentences(tokenize.sent_tokenize(s.replace('\n',''))) for s in summaries]
        del summaries
        sect_sent_array = [merge_sentences(tokenize.sent_tokenize(s.replace('\n',''))) for s in sections]
        del sections
        #1 for every sentence in summary identify the best matching sentence
        scores = []
        for n_summ, summ in enumerate(summ_sent_array):
            for n_summ_sent, sent_summ in enumerate(summ):
                #logging.info('processed summary n: %d, sentence n: %d'  % (n_summ, n_summ_sent))
                for n_sect, sect in enumerate(sect_sent_array):
                    for n_sect_sent, sect_sent in enumerate(sect):
                        # if isinstance(sent_summ, list):
                        #     print(sent_summ)
                        score = self._sentence_score4(scorer, scorer1, sect_sent, sent_summ)
                        scores.append(pd.DataFrame(data=[[n_summ, n_summ_sent, n_sect,
                                                          n_sect_sent, score, len(sect_sent)
                                                          ]],
                                                   columns=['n_summ', 'n_summ_sent',
                                                            'n_sect', 'n_sect_sent', 'score', 'm_words'
                                                            ]
                                                   ))
        del scorer
        del scorer1
        #find max_match for every sentence in summary and tag as matched
        scores = pd.concat(scores, axis=0)
        scores.reset_index(inplace=True)
        scores['matched'] = False
        scores['matched_summ'] = ""
        scores['m_words'] = 1
        gby_summ_sent = scores.groupby(['n_summ', 'n_summ_sent'])
        for key in gby_summ_sent.groups:
            grp = gby_summ_sent.get_group(key)
            max_match = grp['score'].idxmax()
            scores.loc[max_match, 'matched'] = True
            #store the matching sentence in the summary
            scores.loc[max_match, 'matched_summ'] = summ_sent_array[key[0]][key[1]]
            #ket_sent_sect = scores.loc[max_match, ['n_sect', 'n_sect_sent']].values
            #sent_sect = sect_sent_array[ket_sent_sect[0]][ket_sent_sect[1]]
            # logging.info("matched sect %d, sntence %d  ->summ%d, sentence %d" % (ket_sent_sect[0],
            #                                                                             ket_sent_sect[1],
            #                                                                             key[0],
            #                                                                             key[1]
            #                                                                             ) )
            # print('++++++++++++ matched sect %d, sntence %d  ->summ%d, sentence %d' % (ket_sent_sect[0],
            #                                                                            ket_sent_sect[1],
            #                                                                            key[0],
            #                                                                            key[1]
            #                                                                            ) )
            # print(summ_sent_array[key[0]][key[1]])
            # print('best scores, matches sentence-------------------')
            # print(grp.sort_values(['score'])['score'][-4:])
            # print(sent_sect)
            # print('----best 3 candidates after------')

            # for n in grp.sort_values(['score']).index.values[-4:-1]:
            #     print('+++---+++---')
            #     ket_sent_sect2 = scores.loc[n, ['n_sect', 'n_sect_sent']].values
            #     sent_sect2 = sect_sent_array[ket_sent_sect2[0]][ket_sent_sect2[1]]
            #     print(sent_sect2)

            # print('--------------.....---------------')
        #for very section build summary summary
        ###find now best summary
        gby_summ_sent = scores[scores['matched']].groupby(['n_summ'])
        best_summ = 0
        best_score = 0
        for key in gby_summ_sent.groups:
            grp = gby_summ_sent.get_group(key)
            tot_score_summ = np.sum(grp['score'].values) /sum(grp['m_words'].values)
            if tot_score_summ>best_score:
                best_summ = key
                best_score = tot_score_summ
        #write best sumamry to dir
        copyfile(sec_summ_files[1][best_summ],
                 self._out_file_path(os.path.join(self.output_dir,
                                                  self.dir_prefix_tmp + '_best_L_summaries' + str(self.version)),
                                     os.path.basename(sec_summ_files[1][best_summ])
                                     )
                 )
        ##now consider only the best summary and match back sentences
        # last two in sorting are only valid in case of single summary
        scores = scores[scores['n_summ'] == best_summ].sort_values(['n_sect', 'n_sect_sent', 'n_summ_sent'])
        gby_sect_sent = scores[scores['matched']].groupby(['n_sect'])
        for key in gby_sect_sent.groups:
            grp = gby_sect_sent.get_group(key)
            scores = grp['score'].values
            scores_ok = scores>0.0001
            sect_summ = ' '.join(grp['matched_summ'].values[scores_ok])
            if not scores_ok.any():
                continue
            sect_avg_score = np.mean(scores[scores_ok] )
            self._text_to_file(sect_summ,
                               self._out_file_path(os.path.join(self.output_dir,
                                                                self.dir_prefix_tmp+'_summaries'+str(self.version)),
                                                   sec_summ_files[0][key]
                                                   )
                               )
            self._text_to_file(str(sect_avg_score),
                               self._out_file_path(os.path.join(self.output_dir,
                                                                self.dir_prefix_tmp + '_summaries_scores'+str(self.version)),
                                                   sec_summ_files[0][key]
                                                   )
                               )
        logging.info('processed %s' % sec_summ_files[0][0])
        return True

    def predictions_to_files(self, out_doc_name, predictions_out_name, sect_dir ):
        '''this ia matching the document output in one file in th list of section
            out_doc_name: file of original documents out of fine tuning eval on cloud, one per line
            predictions_out_name: file of predicted summaries one per line out of fin tunin eval on cloud
            sect_dir: dir where  out_doc_name content is coming from
        '''
        f_in = open(out_doc_name, "r", encoding="utf-8")
        doc_raw = f_in.readlines()
        self.docs = []
        for line in tqdm(doc_raw):
            self.docs.append(line[:512].split(" "))
        f_in.close()
        del doc_raw
        #list all sections
        files_sect = self._list_file(sect_dir)
        manager = Manager()
        self.match_dict = manager.dict()
        p = Pool(4)
        p.map(self._doc_to_doc, files_sect)
        #now save down predictions in dir with correct name
        f_predict_in = open(predictions_out_name, "r", encoding="utf-8")
        predictions_raw = f_predict_in.readlines()
        out_prediction_dir = os.path.join(os.path.dirname(os.path.normpath(predictions_out_name)), r"sect_predictions" )
        self._create_dir(out_prediction_dir)
        #matched_lines = list(self.match_dict.keys())
        for n, line in enumerate(tqdm(predictions_raw)):
            if n in self.match_dict:
                matched_file = self.match_dict[n]
                self._text_to_file(line.replace("??", ""),
                                   self._out_file_path(out_prediction_dir, matched_file)
                                   )
            else:
                logging.info('missing %s' % (n))

    def _doc_to_doc(self, file_sect):
        '''receives in input a sectiona file and loop through all the documents to find the matching'''
        f_in = open(file_sect, "r", encoding="utf-8")
        sect = f_in.read(512).split(" ")
        f_in.close()
        n_max_match = 0
        for n, doc_start in enumerate(self.docs):
            n_match = sum(1 for w in sect if w in doc_start)
            if n_match>0.975*len(sect):
                n_max_match = n_match
                match = n
                break
            else:
                if n_match>n_max_match:
                    n_max_match = n_match
                    match = n
        #assert(match not in self.match_dict.keys())
        prc_match = n_max_match/len(sect)
        if match not in self.match_dict and prc_match>0.975:
            self.match_dict[match] = os.path.basename(os.path.normpath(file_sect))
            logging.info("matched %s with doc n:%d, prc match: %f " % (file_sect, match, prc_match))

    def score_summary(self, out_summary_dir, gold_summaries, quick=False, min_nwords=0):
        '''score output summary (after predictions_to_files) with gold summaries
        '''
        out_summary_dir = os.path.join(self.root_dir, out_summary_dir)
        #get array of files. every row is one document
        self.quick = quick
        files_docid, outsumm_array = self._doc_files_array(out_summary_dir, narrative_only=False)
        files_gold, gold_array = self._doc_files_array(gold_summaries, narrative_only=False)
        #filter for summary out
        gold_array = [s[1] for s in zip(files_gold, gold_array) if s[0] in files_docid]
        files_gold = files_docid
        #merge section summaries into one doc summary
        references = []
        out_summaries = []
        count = 0
        for files_outsumm, files_gold in tqdm(zip(outsumm_array, gold_array)):
            summary = " ".join([open(f, 'r', encoding ='utf-8').read() for f in files_outsumm])
            gold_summ = [open(f, 'r', encoding ='utf-8').read() for f in files_gold]
            if len(summary.split(" "))>min_nwords and max([len(l.split(" ")) for l in gold_summ])>min_nwords:
                count = count +1
                out_summaries.append(summary)
                references.append(gold_summ)

        logging.info('now computing the metrics... will take some time')
        logging.info('kept %d files, min nwords:%d' % (count, min_nwords))
        logging.info('dir: %s' % (out_summary_dir))
        if self.quick:
            rouge = PyRouge(rouge_n=(1,2), rouge_l=False, rouge_w=True,
                            rouge_w_weight=1.2, rouge_s=False, rouge_su=False,
                            skip_gap=0
                            )
        else:
            rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=True,
                            rouge_w_weight = 1.2, rouge_s = False, rouge_su = False,
                            skip_gap = 0
                            )
        scores = rouge.evaluate(out_summaries, references)
        logging.info("finished...")
        print(json.dumps(scores, indent=2))