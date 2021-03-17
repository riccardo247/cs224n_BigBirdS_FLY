from dataset import Dataset
import os
####step for every dataset
#1) clean_dir called on sections, full_text, and maybe gold summaries

#3) sec_to_summary4  is matching section to its portion of the best summary. it creaates also dir summaries_score with score for every section summary. 5% lowest will be cut
#4) xecute save_tfrecords load_from_dir. will save examples into tfrecors
#5)doc_from_sections mrge all summaries of section together
#6)score_summary compute ROUGE score
def main():
    dataset = Dataset(output_dir=r"D:\stanford\data\FNS summarisation\clean_data_v2",
                      root_dir=r"D:\stanford\data\FNS summarisation\training",
                      clean_deep=False)
    dataset.clean_dir("training\training_sections")
    dataset.clean_dir(r"training\training_gold_standards")
    dataset.clean_dir(r"training\training_full_text")
    dataset.select_best_summary(r'training_gold_standards',
                                r'training_full_text',
                                )
    dataset.sec_to_summary4(r'training_sections',
                          r'training_gold_standards',
                           version=4
                          )


    dataset.predictions_to_files(r'output\summarization_FNS_plargeb_512_8.0_output_validation_decoded_doc_out_1080.txt',
                                 r'output\summarization_FNS_plargeb_512_8.0_output_validation_decoded_predictions_out_1080.txt',
                                 r'validation_sections'
                               )

    os.rename(r"output\sect_predictions",
                            r"output\eval_sect_predictions_ckpt1080_512_v8")
    dataset.doc_from_sections(
        r'output\eval_sect_predictions_ckpt1080_512_v8', clean_deep=False)
    dataset.score_summary(r"output\eval_sect_predictions_ckpt1080_512_v8_merge",
                          r'validation_gold_standards',
                          quick=True,
                          min_nwords=200)


    #dataset.compare_len(r'D:\stanford\data\FNS summarisation\clean_data\training_full_text',
    #                    r'D:\stanford\data\FNS summarisation\clean_data\training_sections_merge'
    #                    )
    #dataset.dir_stats(r'D:\stanford\data\FNS summarisation\training\training\training_sections')
    #dataset.tokenize_file(r'D:\stanford\data\FNS summarisation\training\training\training_sections\4346_225492_0.txt',
    #                      r'C:\Users\ricca\OneDrive\Documenti\stanford\NLP\project\google cloud\bigbird\bigbird\vocab\pegasus.model')
    #dataset.tokenize_dir_stats(r'D:\stanford\data\FNS summarisation\training\training\training_sections',
    #                           r'C:\Users\ricca\OneDrive\Documenti\stanford\NLP\project\google cloud\bigbird\bigbird\vocab\pegasus.model'
    #                           )
    #
    # dataset.count_uniq_doc(r'D:\stanford\data\FNS summarisation\clean_data\training_sections_summaries')
    # dataset.check_missing_summ(r'D:\stanford\data\FNS summarisation\clean_data\training_sections',
    #                         r'D:\stanford\data\FNS summarisation\clean_data\training_sections_summaries'
    #                         )


if __name__ == '__main__':
    main()