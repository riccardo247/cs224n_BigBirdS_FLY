# from official import nlp
# from official.nlp import bert
# import official.nlp.bert.run_classifier
# import official.nlp.data.classifier_data_lib
import tensorflow as tf
import numpy as np
import glob
from multiprocessing import Process, Manager, Pool
import os
import math
from pathlib import Path

class text_to_tfrecord():
    def __init__(self, n_processes, nfiles_intfr, out_dir, split='train', version=3):
        self.n_processes = n_processes
        self.nfiles_intfr = nfiles_intfr
        self.out_dir = out_dir
        self._create_dir(self.out_dir)
        self.split = split
        self.version = version
    def _create_dir(self, output_dir):
        '''check or create new dir'''
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

    def _int64_feature(slf, value):
        "64 bits feature"
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, doc_text, summary_text, filename):
        feature = {
            'document': self._bytes_feature(doc_text),
            'summary': self._bytes_feature(summary_text),
            'filename': self._bytes_feature(filename),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def serialize_classifier_example(self, doc_text, label, filename):
        """serialize on example"""
        feature = {
            'document': self._bytes_feature(doc_text),
            'label': self._int64_feature(label),
            'filename': self._bytes_feature(filename),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def _list_file(self, data_dir):
        '''list all txt files in data_dir'''
        return list(glob.glob(
                            os.path.join(data_dir, "*.txt")
                              ))
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def exclude_low_f_summaries(self, dir_score):
        """exclude worst summaries"""
        #files_summ = [os.path.basename(f) for f in self._list_file(bestsumm_dir)]
        #dir_score = os.path.join(os.path.dirname(bestsumm_dir), "training_sections_summaries_score"+str(self.version))
        score_files = self._list_file(dir_score)
        scores = [float(open(f, "r", encoding="utf-8").read()) for f in score_files]
        sort_idx = np.argsort(scores)
        low_q = int(0.05 * len(sort_idx))
        #doc_ids = [os.path.basename(f).split("_")[0] for f in score_files]
        return [os.path.basename(score_files[n]) for n in sort_idx[:low_q]]

    def load_from_testdir(self, sect_dir,
                      filter_empty=False):
        '''prepare tfrecords from sections and summary. pair (document, summary)'''
        self.filter_empty = filter_empty
        files = list(glob.glob(
            os.path.join(sect_dir, "*.txt")
        ))
        self._create_dir(self.out_dir)

        #files = files[:1000]
        p = Pool(1)#self.n_processes
        files_par = []
        nfiles_intfr = self.nfiles_intfr
        files_nks = self.chunks(files, nfiles_intfr)
        n_files = math.floor(len(files)/nfiles_intfr)
        for n, fs in enumerate(files_nks):
            files_par.append([
                fs, os.path.join(self.out_dir,
                                 'fns_summary_'+self.split+'.tfrecord-{:05d}-of-{:05d}'.format(n, n_files)
                                 )
                ])
        p.map(self._save_tfrecord_test, files_par)

    def _save_tfrecord_test(self, files_par):
        """save fils test"""
        files = files_par[0]
        file_out = files_par[1]
        with tf.io.TFRecordWriter(file_out) as file_writer:
            for f in files:
                file_name = os.path.basename(os.path.normpath(f))

                with open(f, "r", encoding="utf-8") as f_sect:
                    #if files is there there a matched summary, otherwise section was not in summary
                    summ_text = ""
                    sect_text = f_sect.read()
                    empty_cond = len(summ_text) > 20 and len(sect_text) > 20
                    if not self.filter_empty or (self.filter_empty and empty_cond):

                        #if len(sect_text)<100 or len(summ_text)<100: #there are empty sections and no summaries
                        #    continue
                        example = self.serialize_example(sect_text.lower().encode(),
                                                         summ_text.encode(),
                                                         file_name.encode())
                        file_writer.write(example)
        print('processd n %d to %s' % (len(files), file_out))

    def load_classifier_from_dir(self, sect_dir,
                      score_dir = None,
                      filter_empty=True,
                      only_top_summaries=True):
        '''prepare tfrecords from sections and summary. pair (document, summary)'''
        self.filter_empty = filter_empty
        files = list(glob.glob(
            os.path.join(sect_dir, "*.txt")
        ))
        self._create_dir(self.out_dir)
        ##list of files to keep
        self.only_top_summaries=only_top_summaries
        manager = Manager()
        self.text_dict = manager.dict()
        if score_dir:
            self.files_toexclude = self.exclude_low_f_summaries(score_dir)

        p = Pool(1) #self.n_processes
        files_par = []
        nfiles_intfr = self.nfiles_intfr
        files_nks = self.chunks(files, nfiles_intfr)
        n_files = math.floor(len(files)/nfiles_intfr)
        for n, fs in enumerate(files_nks):
            files_par.append([
                fs, os.path.join(self.out_dir,
                                 'fns_summary_class_'+self.split+'.tfrecord-{:05d}-of-{:05d}'.format(n, n_files)
                                 )
                ])
        p.map(self._save_classifier_tfrecord, files_par)
        print('total n examples: %d' % len(self.text_dict))

    def _save_classifier_tfrecord(self, files_par):
        """save tfrcords files"""
        files = files_par[0]
        file_out = files_par[1]
        with tf.io.TFRecordWriter(file_out) as file_writer:
            for f in files:
                file_name = os.path.basename(os.path.normpath(f))
                keep_it = True
                if keep_it:
                    with open(f, "r", encoding="utf-8") as f_sect:
                        #if files is there there a matched summary, otherwise section was not in summary
                        label = int(file_name.split(".")[0].split("_")[2])>0
                        sect_text = f_sect.read()
                        empty_cond = len(sect_text.split(" ")) > 200
                        if not self.filter_empty or (self.filter_empty and empty_cond):
                            self.text_dict[f] = 1
                            #if len(sect_text)<100 or len(summ_text)<100: #there are empty sections and no summaries
                            #    continue
                            example = self.serialize_classifier_example(sect_text.lower().encode(),
                                                             label,
                                                             file_name.encode())
                            file_writer.write(example)
        print('processd n %d to %s' % (len(files), file_out))

    def load_from_dir(self, sect_dir,
                      summary_dir,
                      score_dir=None,
                      filter_empty=True,
                      only_top_summaries=True,
                      narrative_only=True):
        '''prepare tfrecords from sections and summary. pair (document, summary)'''
        self.filter_empty = filter_empty
        files = list(glob.glob(
            os.path.join(sect_dir, "*.txt")
        ))
        self._create_dir(self.out_dir)
        ##list of files to keep
        self.only_top_summaries = only_top_summaries
        manager = Manager()
        self.text_dict = manager.dict()
        if score_dir:
            self.files_toexclude = self.exclude_low_f_summaries(score_dir)
        # files = files[:1000]
        self.summary_dir = summary_dir
        p = Pool(1)  # self.n_processes
        if narrative_only:
            files = [f for f in files if int(os.path.basename(os.path.splitext(f)[0]).split("_")[2]) > 0]
        files_par = []
        nfiles_intfr = self.nfiles_intfr
        files_nks = self.chunks(files, nfiles_intfr)
        n_files = math.floor(len(files) / nfiles_intfr)
        for n, fs in enumerate(files_nks):
            files_par.append([
                fs, os.path.join(self.out_dir,
                                 'fns_summary_' + self.split + '.tfrecord-{:05d}-of-{:05d}'.format(n, n_files)
                                 )
            ])
        p.map(self._save_tfrecord, files_par)
        print('total n examples: %d' % len(self.text_dict))

    def _save_tfrecord(self, files_par):
        """intrla function to save on file"""
        files = files_par[0]
        file_out = files_par[1]
        with tf.io.TFRecordWriter(file_out) as file_writer:
            for f in files:
                file_name = os.path.basename(os.path.normpath(f))
                summ_name = os.path.join(self.summary_dir,
                                         file_name
                                         )
                keep_it = (not self.only_top_summaries) or \
                          (self.only_top_summaries and os.path.basename(summ_name) not in self.files_toexclude)
                if keep_it:
                    with open(f, "r", encoding="utf-8") as f_sect:
                        # if files is there there a matched summary, otherwise section was not in summary
                        if os.path.isfile(summ_name):
                            f_summ = open(summ_name, "r", encoding="utf-8")
                            summ_text = f_summ.read()
                            f_summ.close()
                        else:
                            summ_text = ""
                        sect_text = f_sect.read()
                        empty_cond = len(summ_text.split(" ")) > 250 and len(sect_text.split(" ")) > 250
                        if not self.filter_empty or (self.filter_empty and empty_cond):
                            self.text_dict[f] = 1
                            # if len(sect_text)<100 or len(summ_text)<100: #there are empty sections and no summaries
                            #    continue
                            example = self.serialize_example(sect_text.lower().encode(),
                                                             summ_text.lower().encode(),
                                                             file_name.encode())
                            file_writer.write(example)
        print('processd n %d to %s' % (len(files), file_out))

def main():
    ###normal save file
    rootdir = 'D:\stanford\data\FNS summarisation\clean_data_v2'
    txt_to_tfr = text_to_tfrecord(n_processes=5,
                                  nfiles_intfr=4000,
                                  out_dir = rootdir+r'tfds\fns_summary\6.0.0',
                                  split='train',
                                  version=6)
    txt_to_tfr.load_from_dir(
                            sect_dir = rootdir+r'training_sections',
                            summary_dir = rootdir+r'training_sections_summaries4',
                            score_dir = rootdir+r'training_sections_summaries_scores4',
                            )

if __name__ == "__main__":
    main()