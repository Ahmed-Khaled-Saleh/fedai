# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/fedai',
                'doc_host': 'https://Ahmed-Khaled-Saleh.github.io',
                'git_url': 'https://github.com/Ahmed-Khaled-Saleh/fedai',
                'lib_path': 'fedai'},
  'syms': { 'fedai.FLearner': { 'fedai.FLearner.FLearner': ('flearner.html#flearner', 'fedai/FLearner.py'),
                                'fedai.FLearner.FLearner.__init__': ('flearner.html#flearner.__init__', 'fedai/FLearner.py'),
                                'fedai.FLearner.FLearner.run_simulation': ('flearner.html#flearner.run_simulation', 'fedai/FLearner.py'),
                                'fedai.FLearner.client_fn': ('flearner.html#client_fn', 'fedai/FLearner.py')},
            'fedai.client_selector': { 'fedai.client_selector.BaseClientSelector': ( 'client_selection.html#baseclientselector',
                                                                                     'fedai/client_selector.py'),
                                       'fedai.client_selector.BaseClientSelector.__init__': ( 'client_selection.html#baseclientselector.__init__',
                                                                                              'fedai/client_selector.py'),
                                       'fedai.client_selector.BaseClientSelector.select': ( 'client_selection.html#baseclientselector.select',
                                                                                            'fedai/client_selector.py')},
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
            'fedai.data.VisionBlock': { 'fedai.data.VisionBlock.VisionBlock': ( 'data.tensorf.html#visionblock',
                                                                                'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.__getitem__': ( 'data.tensorf.html#visionblock.__getitem__',
                                                                                            'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.__init__': ( 'data.tensorf.html#visionblock.__init__',
                                                                                         'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.__len__': ( 'data.tensorf.html#visionblock.__len__',
                                                                                        'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.download_data': ( 'data.tensorf.html#visionblock.download_data',
                                                                                              'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.load_single_client_data': ( 'data.tensorf.html#visionblock.load_single_client_data',
                                                                                                        'fedai/data/VisionBlock.py'),
                                        'fedai.data.VisionBlock.VisionBlock.tensorify': ( 'data.tensorf.html#visionblock.tensorify',
                                                                                          'fedai/data/VisionBlock.py')},
            'fedai.data.core': { 'fedai.data.core.BaseDownloader': ('data.core.html#basedownloader', 'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.__init__': ( 'data.core.html#basedownloader.__init__',
                                                                              'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.check': ('data.core.html#basedownloader.check', 'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.load_data': ( 'data.core.html#basedownloader.load_data',
                                                                               'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.load_partition': ( 'data.core.html#basedownloader.load_partition',
                                                                                    'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.load_split': ( 'data.core.html#basedownloader.load_split',
                                                                                'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.partition': ( 'data.core.html#basedownloader.partition',
                                                                               'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.save_partitions': ( 'data.core.html#basedownloader.save_partitions',
                                                                                     'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.save_partitions_np': ( 'data.core.html#basedownloader.save_partitions_np',
                                                                                        'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.save_space': ( 'data.core.html#basedownloader.save_space',
                                                                                'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.split_data': ( 'data.core.html#basedownloader.split_data',
                                                                                'fedai/data/core.py'),
                                 'fedai.data.core.BaseDownloader.tensorify': ( 'data.core.html#basedownloader.tensorify',
                                                                               'fedai/data/core.py'),
                                 'fedai.data.core.LLMDataCollator': ('data.core.html#llmdatacollator', 'fedai/data/core.py')},
            'fedai.data.downloader': { 'fedai.data.downloader.BaseDownloader': ( 'data.downloader.html#basedownloader',
                                                                                 'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.__init__': ( 'data.downloader.html#basedownloader.__init__',
                                                                                          'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.check': ( 'data.downloader.html#basedownloader.check',
                                                                                       'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.load_data': ( 'data.downloader.html#basedownloader.load_data',
                                                                                           'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.partition': ( 'data.downloader.html#basedownloader.partition',
                                                                                           'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.save_partitions': ( 'data.downloader.html#basedownloader.save_partitions',
                                                                                                 'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.save_space': ( 'data.downloader.html#basedownloader.save_space',
                                                                                            'fedai/data/downloader.py'),
                                       'fedai.data.downloader.BaseDownloader.split_data': ( 'data.downloader.html#basedownloader.split_data',
                                                                                            'fedai/data/downloader.py'),
                                       'fedai.data.downloader.LLMDataCollator': ( 'data.downloader.html#llmdatacollator',
                                                                                  'fedai/data/downloader.py')},
            'fedai.data.partitioners': { 'fedai.data.partitioners.BasePartitioner': ( 'data.partitioners.html#basepartitioner',
                                                                                      'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.__init__': ( 'data.partitioners.html#basepartitioner.__init__',
                                                                                               'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.assign': ( 'data.partitioners.html#basepartitioner.assign',
                                                                                             'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.BasePartitioner.partition': ( 'data.partitioners.html#basepartitioner.partition',
                                                                                                'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.DirPartitioner': ( 'data.partitioners.html#dirpartitioner',
                                                                                     'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.DirPartitioner.__init__': ( 'data.partitioners.html#dirpartitioner.__init__',
                                                                                              'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.DirPartitioner.partition': ( 'data.partitioners.html#dirpartitioner.partition',
                                                                                               'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.ExtPartitioner': ( 'data.partitioners.html#extpartitioner',
                                                                                     'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.ExtPartitioner.__init__': ( 'data.partitioners.html#extpartitioner.__init__',
                                                                                              'fedai/data/partitioners.py'),
                                         'fedai.data.partitioners.ExtPartitioner.partition': ( 'data.partitioners.html#extpartitioner.partition',
                                                                                               'fedai/data/partitioners.py')},
            'fedai.data.tensorf': { 'fedai.data.tensorf.VisionBlock': ('data.tensorf.html#visionblock', 'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.__getitem__': ( 'data.tensorf.html#visionblock.__getitem__',
                                                                                    'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.__init__': ( 'data.tensorf.html#visionblock.__init__',
                                                                                 'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.__len__': ( 'data.tensorf.html#visionblock.__len__',
                                                                                'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.download_data': ( 'data.tensorf.html#visionblock.download_data',
                                                                                      'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.load_single_client_data': ( 'data.tensorf.html#visionblock.load_single_client_data',
                                                                                                'fedai/data/tensorf.py'),
                                    'fedai.data.tensorf.VisionBlock.tensorify': ( 'data.tensorf.html#visionblock.tensorify',
                                                                                  'fedai/data/tensorf.py')},
            'fedai.federated.agents': { 'fedai.federated.agents.Agent': ('federated.agents.html#agent', 'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.__init__': ( 'federated.agents.html#agent.__init__',
                                                                                   'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.clear_model': ( 'federated.agents.html#agent.clear_model',
                                                                                      'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.communicate': ( 'federated.agents.html#agent.communicate',
                                                                                      'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.init_agent': ( 'federated.agents.html#agent.init_agent',
                                                                                     'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.save_state': ( 'federated.agents.html#agent.save_state',
                                                                                     'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Agent.update_state': ( 'federated.agents.html#agent.update_state',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.AgentMira': ( 'federated.agents.html#agentmira',
                                                                              'fedai/federated/agents.py'),
                                        'fedai.federated.agents.AgentMira.__init__': ( 'federated.agents.html#agentmira.__init__',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.AgentRole': ( 'federated.agents.html#agentrole',
                                                                              'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent': ('federated.agents.html#flagent', 'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.__init__': ( 'federated.agents.html#flagent.__init__',
                                                                                     'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.__str__': ( 'federated.agents.html#flagent.__str__',
                                                                                    'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent._closure': ( 'federated.agents.html#flagent._closure',
                                                                                     'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent._forward': ( 'federated.agents.html#flagent._forward',
                                                                                     'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent._run_batch': ( 'federated.agents.html#flagent._run_batch',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent._run_epoch': ( 'federated.agents.html#flagent._run_epoch',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.aggregate': ( 'federated.agents.html#flagent.aggregate',
                                                                                      'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.clear_model': ( 'federated.agents.html#flagent.clear_model',
                                                                                        'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.fit': ( 'federated.agents.html#flagent.fit',
                                                                                'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.get_batch': ( 'federated.agents.html#flagent.get_batch',
                                                                                      'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.runFL': ( 'federated.agents.html#flagent.runfl',
                                                                                  'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.save_state': ( 'federated.agents.html#flagent.save_state',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.server_init': ( 'federated.agents.html#flagent.server_init',
                                                                                        'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FLAgent.test': ( 'federated.agents.html#flagent.test',
                                                                                 'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FedSophiaAgent': ( 'federated.agents.html#fedsophiaagent',
                                                                                   'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FedSophiaAgent.__init__': ( 'federated.agents.html#fedsophiaagent.__init__',
                                                                                            'fedai/federated/agents.py'),
                                        'fedai.federated.agents.FedSophiaAgent.train': ( 'federated.agents.html#fedsophiaagent.train',
                                                                                         'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Fedu': ('federated.agents.html#fedu', 'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Fedu.__init__': ( 'federated.agents.html#fedu.__init__',
                                                                                  'fedai/federated/agents.py'),
                                        'fedai.federated.agents.Fedu.aggregate': ( 'federated.agents.html#fedu.aggregate',
                                                                                   'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PadgAgent': ( 'federated.agents.html#padgagent',
                                                                              'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PadgAgent.__init__': ( 'federated.agents.html#padgagent.__init__',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PadgAgent.aggregate': ( 'federated.agents.html#padgagent.aggregate',
                                                                                        'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PadgAgent.apply_constraints': ( 'federated.agents.html#padgagent.apply_constraints',
                                                                                                'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PadgAgent.compute_probs': ( 'federated.agents.html#padgagent.compute_probs',
                                                                                            'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PeftAgent': ( 'federated.agents.html#peftagent',
                                                                              'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PeftAgent.__init__': ( 'federated.agents.html#peftagent.__init__',
                                                                                       'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PeftAgent.init_agent': ( 'federated.agents.html#peftagent.init_agent',
                                                                                         'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PeftAgent.peftify': ( 'federated.agents.html#peftagent.peftify',
                                                                                      'fedai/federated/agents.py'),
                                        'fedai.federated.agents.PeftAgent.save_state_': ( 'federated.agents.html#peftagent.save_state_',
                                                                                          'fedai/federated/agents.py')},
            'fedai.learner_utils': { 'fedai.learner_utils.get_block': ('learner_utils.html#get_block', 'fedai/learner_utils.py'),
                                     'fedai.learner_utils.get_cls': ('learner_utils.html#get_cls', 'fedai/learner_utils.py'),
                                     'fedai.learner_utils.get_criterion': ('learner_utils.html#get_criterion', 'fedai/learner_utils.py'),
                                     'fedai.learner_utils.get_model': ('learner_utils.html#get_model', 'fedai/learner_utils.py'),
                                     'fedai.learner_utils.load_state_from_disk': ( 'learner_utils.html#load_state_from_disk',
                                                                                   'fedai/learner_utils.py')},
            'fedai.main': {},
            'fedai.metrics': { 'fedai.metrics.Metrics': ('metrics.html#metrics', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.__init__': ('metrics.html#metrics.__init__', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.compute': ('metrics.html#metrics.compute', 'fedai/metrics.py'),
                               'fedai.metrics.Metrics.prepare_targets_llm': ( 'metrics.html#metrics.prepare_targets_llm',
                                                                              'fedai/metrics.py')},
            'fedai.models': { 'fedai.models.LogisticRegression': ('models.html#logisticregression', 'fedai/models.py'),
                              'fedai.models.LogisticRegression.__init__': ('models.html#logisticregression.__init__', 'fedai/models.py'),
                              'fedai.models.LogisticRegression.forward': ('models.html#logisticregression.forward', 'fedai/models.py'),
                              'fedai.models.MLP': ('models.html#mlp', 'fedai/models.py'),
                              'fedai.models.MLP.__init__': ('models.html#mlp.__init__', 'fedai/models.py'),
                              'fedai.models.MLP.forward': ('models.html#mlp.forward', 'fedai/models.py')},
            'fedai.optimizers': { 'fedai.optimizers.SophiaG': ('optimizers.html#sophiag', 'fedai/optimizers.py'),
                                  'fedai.optimizers.SophiaG.__init__': ('optimizers.html#sophiag.__init__', 'fedai/optimizers.py'),
                                  'fedai.optimizers.SophiaG.__setstate__': ('optimizers.html#sophiag.__setstate__', 'fedai/optimizers.py'),
                                  'fedai.optimizers.SophiaG.step': ('optimizers.html#sophiag.step', 'fedai/optimizers.py'),
                                  'fedai.optimizers.SophiaG.update_hessian': ( 'optimizers.html#sophiag.update_hessian',
                                                                               'fedai/optimizers.py'),
                                  'fedai.optimizers._single_tensor_sophiag': ( 'optimizers.html#_single_tensor_sophiag',
                                                                               'fedai/optimizers.py'),
                                  'fedai.optimizers.sophiag': ('optimizers.html#sophiag', 'fedai/optimizers.py')},
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
                                                                             'fedai/trainers.py')},
            'fedai.utils': { 'fedai.utils.draw_matrix': ('utils.ipynb.html#draw_matrix', 'fedai/utils.py'),
                             'fedai.utils.draw_nx_graph': ('utils.ipynb.html#draw_nx_graph', 'fedai/utils.py'),
                             'fedai.utils.generate_graph': ('utils.ipynb.html#generate_graph', 'fedai/utils.py'),
                             'fedai.utils.get_class': ('utils.ipynb.html#get_class', 'fedai/utils.py'),
                             'fedai.utils.get_server': ('utils.ipynb.html#get_server', 'fedai/utils.py'),
                             'fedai.utils.load_config': ('utils.ipynb.html#load_config', 'fedai/utils.py'),
                             'fedai.utils.prepare_dl': ('utils.ipynb.html#prepare_dl', 'fedai/utils.py'),
                             'fedai.utils.save_space': ('utils.ipynb.html#save_space', 'fedai/utils.py')},
            'fedai.vision.VisionBlock': { 'fedai.vision.VisionBlock.VisionBlock': ( 'vision.visionblock.html#visionblock',
                                                                                    'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.__getitem__': ( 'vision.visionblock.html#visionblock.__getitem__',
                                                                                                'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.__init__': ( 'vision.visionblock.html#visionblock.__init__',
                                                                                             'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.__len__': ( 'vision.visionblock.html#visionblock.__len__',
                                                                                            'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.download_data': ( 'vision.visionblock.html#visionblock.download_data',
                                                                                                  'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.load_single_client_data': ( 'vision.visionblock.html#visionblock.load_single_client_data',
                                                                                                            'fedai/vision/VisionBlock.py'),
                                          'fedai.vision.VisionBlock.VisionBlock.tensorify': ( 'vision.visionblock.html#visionblock.tensorify',
                                                                                              'fedai/vision/VisionBlock.py')},
            'fedai.vision.downloader': { 'fedai.vision.downloader.VisionDownloader': ( 'vision.downloader.html#visiondownloader',
                                                                                       'fedai/vision/downloader.py'),
                                         'fedai.vision.downloader.VisionDownloader.__init__': ( 'vision.downloader.html#visiondownloader.__init__',
                                                                                                'fedai/vision/downloader.py'),
                                         'fedai.vision.downloader.VisionDownloader.load_data': ( 'vision.downloader.html#visiondownloader.load_data',
                                                                                                 'fedai/vision/downloader.py')},
            'fedai.vision.models': { 'fedai.vision.models.CIFAR10CNN': ('vision.models.html#cifar10cnn', 'fedai/vision/models.py'),
                                     'fedai.vision.models.CIFAR10CNN.__init__': ( 'vision.models.html#cifar10cnn.__init__',
                                                                                  'fedai/vision/models.py'),
                                     'fedai.vision.models.CIFAR10CNN.forward': ( 'vision.models.html#cifar10cnn.forward',
                                                                                 'fedai/vision/models.py'),
                                     'fedai.vision.models.MNISTCNN': ('vision.models.html#mnistcnn', 'fedai/vision/models.py'),
                                     'fedai.vision.models.MNISTCNN.__init__': ( 'vision.models.html#mnistcnn.__init__',
                                                                                'fedai/vision/models.py'),
                                     'fedai.vision.models.MNISTCNN.forward': ( 'vision.models.html#mnistcnn.forward',
                                                                               'fedai/vision/models.py')},
            'fedai.wandb_writer': { 'fedai.wandb_writer.WandbWriter': ('wandb_writer.html#wandbwriter', 'fedai/wandb_writer.py'),
                                    'fedai.wandb_writer.WandbWriter.__init__': ( 'wandb_writer.html#wandbwriter.__init__',
                                                                                 'fedai/wandb_writer.py'),
                                    'fedai.wandb_writer.WandbWriter.finish': ( 'wandb_writer.html#wandbwriter.finish',
                                                                               'fedai/wandb_writer.py'),
                                    'fedai.wandb_writer.WandbWriter.save': ('wandb_writer.html#wandbwriter.save', 'fedai/wandb_writer.py'),
                                    'fedai.wandb_writer.WandbWriter.write': ( 'wandb_writer.html#wandbwriter.write',
                                                                              'fedai/wandb_writer.py')}}}
