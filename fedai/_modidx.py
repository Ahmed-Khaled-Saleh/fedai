# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/fedai',
                'doc_host': 'https://Ahmed-Khaled-Saleh.github.io',
                'git_url': 'https://github.com/Ahmed-Khaled-Saleh/fedai',
                'lib_path': 'fedai'},
  'syms': { 'fedai.FLearner': {},
            'fedai.clients': { 'fedai.clients.BaseClient': ('clients.html#baseclient', 'fedai/clients.py'),
                               'fedai.clients.BaseClient.__init__': ('clients.html#baseclient.__init__', 'fedai/clients.py'),
                               'fedai.clients.Client_mira': ('clients.html#client_mira', 'fedai/clients.py'),
                               'fedai.clients.Client_mira.__init__': ('clients.html#client_mira.__init__', 'fedai/clients.py'),
                               'fedai.clients.Client_mira.clear_model': ('clients.html#client_mira.clear_model', 'fedai/clients.py'),
                               'fedai.clients.Client_mira.init_local_train': ( 'clients.html#client_mira.init_local_train',
                                                                               'fedai/clients.py'),
                               'fedai.clients.Client_mira.terminate_local_train': ( 'clients.html#client_mira.terminate_local_train',
                                                                                    'fedai/clients.py'),
                               'fedai.clients.Client_mira.train': ('clients.html#client_mira.train', 'fedai/clients.py')},
            'fedai.core': { 'fedai.core.CommunicationSystem': ('core.html#communicationsystem', 'fedai/core.py'),
                            'fedai.core.FLSystem': ('core.html#flsystem', 'fedai/core.py'),
                            'fedai.core.get_cfg': ('core.html#get_cfg', 'fedai/core.py'),
                            'fedai.core.say_hello': ('core.html#say_hello', 'fedai/core.py')},
            'fedai.data.core': { 'fedai.data.core.FDownloader': ('data.core.html#fdownloader', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.__init__': ('data.core.html#fdownloader.__init__', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.check': ('data.core.html#fdownloader.check', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.load_data': ('data.core.html#fdownloader.load_data', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.load_partition': ( 'data.core.html#fdownloader.load_partition',
                                                                                 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.load_split': ('data.core.html#fdownloader.load_split', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.partition': ('data.core.html#fdownloader.partition', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.save_partitions': ( 'data.core.html#fdownloader.save_partitions',
                                                                                  'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.save_partitions_np': ( 'data.core.html#fdownloader.save_partitions_np',
                                                                                     'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.split_data': ('data.core.html#fdownloader.split_data', 'fedai/data/core.py'),
                                 'fedai.data.core.FDownloader.tensorify': ('data.core.html#fdownloader.tensorify', 'fedai/data/core.py'),
                                 'fedai.data.core.LLMDataCollator': ('data.core.html#llmdatacollator', 'fedai/data/core.py')},
            'fedai.data.partitioners': { 'fedai.data.partitioners.BasePartitioner': ( 'data.partitioners.html#basepartitioner',
                                                                                      'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.__init__': ( 'data.partitioners.html#basepartitioner.__init__',
                                                                                               'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.assign': ( 'data.partitioners.html#basepartitioner.assign',
                                                                                             'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.parition': ( 'data.partitioners.html#basepartitioner.parition',
                                                                                               'fedai/data/partitioners.py')},
            'fedai.data.tensorf': { 'fedai.data.tensorf.TensorF': ('data.tensorf.html#tensorf', 'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.TensorF.__getitem__': ( 'data.tensorf.html#tensorf.__getitem__',
                                                                                'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.TensorF.__init__': ('data.tensorf.html#tensorf.__init__', 'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.TensorF.__len__': ('data.tensorf.html#tensorf.__len__', 'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.TensorF.load_single_client_data': ( 'data.tensorf.html#tensorf.load_single_client_data',
                                                                                            'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.TensorF.tensorify': ( 'data.tensorf.html#tensorf.tensorify',
                                                                              'fedai/data/tensorf.py')},
            'fedai.metrics': { 'fedai.metrics.Metrics': ('metrics.html#metrics', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.__init__': ('metrics.html#metrics.__init__', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.compute': ('metrics.html#metrics.compute', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.prepare_targets_llm': ( 'metrics.html#metrics.prepare_targets_llm',
                                                                              'fedai/metrics.py')},
            'fedai.models': { 'fedai.models.LogisticRegression': ('models.ipynb.html#logisticregression', 'fedai/models.py'),
                              'fedai.models.LogisticRegression.__init__': ( 'models.ipynb.html#logisticregression.__init__',
                                                                            'fedai/models.py'),
                              'fedai.models.LogisticRegression.forward': ( 'models.ipynb.html#logisticregression.forward',
                                                                           'fedai/models.py'),
                              'fedai.models.get_model': ('models.ipynb.html#get_model', 'fedai/models.py')},
            'fedai.servers': { 'fedai.servers.BaseServer': ('servers.html#baseserver', 'fedai/servers.py'),
                               'fedai.servers.BaseServer.__init__': ('servers.html#baseserver.__init__', 'fedai/servers.py'),
                               'fedai.servers.BaseServer.__str__': ('servers.html#baseserver.__str__', 'fedai/servers.py'),
                               'fedai.servers.BaseServer.client_selection': ( 'servers.html#baseserver.client_selection',
                                                                              'fedai/servers.py'),
                               'fedai.servers.BaseServer.get_selected_client': ( 'servers.html#baseserver.get_selected_client',
                                                                                 'fedai/servers.py'),
                               'fedai.servers.BaseServer.send': ('servers.html#baseserver.send', 'fedai/servers.py'),
                               'fedai.servers.Server_mira': ('servers.html#server_mira', 'fedai/servers.py'),
                               'fedai.servers.Server_mira.__init__': ('servers.html#server_mira.__init__', 'fedai/servers.py'),
                               'fedai.servers.Server_mira.aggregate': ('servers.html#server_mira.aggregate', 'fedai/servers.py'),
                               'fedai.servers.Server_mira.init_sim_matrix': ( 'servers.html#server_mira.init_sim_matrix',
                                                                              'fedai/servers.py'),
                               'fedai.servers.Server_mira.update': ('servers.html#server_mira.update', 'fedai/servers.py')},
            'fedai.text.data': { 'fedai.text.data.DefaultToken': ('text.data.html#defaulttoken', 'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet': ('text.data.html#mtldataset', 'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet.__getitem__': ('text.data.html#mtldataset.__getitem__', 'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet.__init__': ('text.data.html#mtldataset.__init__', 'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet.__len__': ('text.data.html#mtldataset.__len__', 'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet._tokenize_fn': ( 'text.data.html#mtldataset._tokenize_fn',
                                                                              'fedai/text/data.py'),
                                 'fedai.text.data.MTLDataSet.preprocess': ('text.data.html#mtldataset.preprocess', 'fedai/text/data.py')},
            'fedai.text.models': { 'fedai.text.models.CausalLMModel': ('text.models.html#causallmmodel', 'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMModel.__init__': ( 'text.models.html#causallmmodel.__init__',
                                                                                 'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMModel.forward': ( 'text.models.html#causallmmodel.forward',
                                                                                'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMPEFTModel': ('text.models.html#causallmpeftmodel', 'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMPEFTModel.__getattr__': ( 'text.models.html#causallmpeftmodel.__getattr__',
                                                                                        'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMPEFTModel.__init__': ( 'text.models.html#causallmpeftmodel.__init__',
                                                                                     'fedai/text/models.py'),
                                   'fedai.text.models.CausalLMPEFTModel.forward': ( 'text.models.html#causallmpeftmodel.forward',
                                                                                    'fedai/text/models.py'),
                                   'fedai.text.models.CharacterLSTM': ('text.models.html#characterlstm', 'fedai/text/models.py'),
                                   'fedai.text.models.CharacterLSTM.__init__': ( 'text.models.html#characterlstm.__init__',
                                                                                 'fedai/text/models.py'),
                                   'fedai.text.models.CharacterLSTM.forward': ( 'text.models.html#characterlstm.forward',
                                                                                'fedai/text/models.py'),
                                   'fedai.text.models.CharacterLSTM.init_hidden': ( 'text.models.html#characterlstm.init_hidden',
                                                                                    'fedai/text/models.py'),
                                   'fedai.text.models.delegate': ('text.models.html#delegate', 'fedai/text/models.py'),
                                   'fedai.text.models.get_hf_model': ('text.models.html#get_hf_model', 'fedai/text/models.py')},
            'fedai.trainers': { 'fedai.trainers.LLMTrainer': ('trainers.ipynb.html#llmtrainer', 'fedai/trainers.py'),
                                'fedai.trainers.LLMTrainer.__init__': ('trainers.ipynb.html#llmtrainer.__init__', 'fedai/trainers.py'),
                                'fedai.trainers.LLMTrainer._forward': ('trainers.ipynb.html#llmtrainer._forward', 'fedai/trainers.py'),
                                'fedai.trainers.LLMTrainer.get_batch': ('trainers.ipynb.html#llmtrainer.get_batch', 'fedai/trainers.py'),
                                'fedai.trainers.LLMTrainer.test_generate': ( 'trainers.ipynb.html#llmtrainer.test_generate',
                                                                             'fedai/trainers.py'),
                                'fedai.trainers.Trainer': ('trainers.ipynb.html#trainer', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer.__init__': ('trainers.ipynb.html#trainer.__init__', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer._closure': ('trainers.ipynb.html#trainer._closure', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer._forward': ('trainers.ipynb.html#trainer._forward', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer._run_batch': ('trainers.ipynb.html#trainer._run_batch', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer._run_epoch': ('trainers.ipynb.html#trainer._run_epoch', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer.get_batch': ('trainers.ipynb.html#trainer.get_batch', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer.test': ('trainers.ipynb.html#trainer.test', 'fedai/trainers.py'),
                                'fedai.trainers.Trainer.train': ('trainers.ipynb.html#trainer.train', 'fedai/trainers.py')},
            'fedai.utils': { 'fedai.utils.LazyList': ('utils.ipynb.html#lazylist', 'fedai/utils.py'),
                             'fedai.utils.LazyList.__getitem__': ('utils.ipynb.html#lazylist.__getitem__', 'fedai/utils.py'),
                             'fedai.utils.LazyList.__init__': ('utils.ipynb.html#lazylist.__init__', 'fedai/utils.py'),
                             'fedai.utils.LazyList.clear_cache': ('utils.ipynb.html#lazylist.clear_cache', 'fedai/utils.py'),
                             'fedai.utils.get_class': ('utils.ipynb.html#get_class', 'fedai/utils.py'),
                             'fedai.utils.get_server': ('utils.ipynb.html#get_server', 'fedai/utils.py'),
                             'fedai.utils.load_config': ('utils.ipynb.html#load_config', 'fedai/utils.py'),
                             'fedai.utils.load_ds': ('utils.ipynb.html#load_ds', 'fedai/utils.py'),
                             'fedai.utils.prepare_dl': ('utils.ipynb.html#prepare_dl', 'fedai/utils.py'),
                             'fedai.utils.save_space': ('utils.ipynb.html#save_space', 'fedai/utils.py')},
            'fedai.vision.data': { 'fedai.vision.data.FVDownloader.load_data': ( 'vision.data.html#fvdownloader.load_data',
                                                                                 'fedai/vision/data.py')},
            'fedai.vision.models': {}}}
