import sys, getopt

import tensorflow as tf

usage_str = 'python rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run' \
            '--out_dir_name=path/to/ckpt_dir_and_name/'


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run, out_dir_name):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
         
            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)
                

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            #saver.save(sess, '/home/iis/Documents/mobilenet_project/merged_ckpt/MobilenetV1_M/MobilenetV1_M.ckpt')
            #saver.save(sess, '/home/iis/Documents/mobilenet_project/merged_ckpt/MobilenetV1_M_1/MobilenetV1_M_1.ckpt')
            saver.save(sess, out_dir_name)

def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False
    out_dir_name = './model.ckpt'

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run=', 'out_dir_name='])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True
        elif opt == '--out_dir_name':
            out_dir_name = arg

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run, out_dir_name )


if __name__ == '__main__':
    main(sys.argv[1:])