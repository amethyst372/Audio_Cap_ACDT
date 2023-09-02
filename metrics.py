from eval_metrics import evaluate_metrics_from_lists, combine_single_and_per_file_metrics, write_json
from datetime import datetime
import itertools

def aac_metrics(outputs, tokenizer):
    all_gt_captions = []
    all_pred_captions = []
    
    current_time = datetime.now().strftime("%y%m%H%M%S")
    
    gt_captions = []
    pred_captions = []
    for i_ex in range(outputs['predictions'].shape[0]):
        gt_ = tokenizer.decode(outputs['label_ids'][i_ex,:])
        gt_captions.append(gt_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        pred_ = tokenizer.decode(outputs['predictions'][i_ex,:])
        pred_captions.append(pred_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        
        # Group for COCO metrics
        if i_ex == len(outputs['filenames'])-1 or outputs['filenames'][i_ex+1] != outputs['filenames'][i_ex]: # Last example for current audio
            # assert(all(x == pred_captions[0] for x in pred_captions))
            all_gt_captions.append(gt_captions)
            all_pred_captions.append(pred_captions[0])
            
            #print('----------')
            #print('Pred: '+pred_captions[0])
            #print('GTs:')
            #for i_gt in range(len(gt_captions)):
            #    print('      '+gt_captions[i_gt])
            
            gt_captions = []
            pred_captions = []
            
    # with open('/home/qust-011/caption/dep_baseline/cap_out/' + current_time + '_gt.txt', 'w') as f:
    #     flt = list(itertools.chain(*all_gt_captions))
    #     f.write('\n'.join(flt))
    with open('/home/qust-011/caption/dep_baseline/cap_out/' + current_time + '_abl.txt', 'w') as f:
        # flt = list(itertools.chain(*all_pred_captions))
        for i in range(len(all_pred_captions)):
            f.write('GT:\n' + '\n'.join(all_gt_captions[i]) + '\n')
            f.write('PRED:\n' + all_pred_captions[i] + '\n')
    
    metrics, per_file_metrics = evaluate_metrics_from_lists(all_pred_captions, all_gt_captions)
    
    file_names = ['{:05d}'.format(i_file) for i_file in range(len(all_gt_captions))]
    
    total_metrics = combine_single_and_per_file_metrics(
        metrics, per_file_metrics, file_names
    )
    
    return {
        key.lower(): value for key, value in total_metrics.items()
    }, all_gt_captions, all_pred_captions
    
