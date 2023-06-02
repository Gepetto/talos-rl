# Tensorboard Integration

To use tensorboard on pfcalul:

- Port 6006 of the server need to be forwarded as follows:

```bash
ssh -L 16006:127.0.0.1:6006 $USER@pfcalcul.laas.fr
```

- Then tensorboard can be run directly on the server:

```bash
cd /pfcalcul/work/cperrot
source tensorboard/bin/activate # Activate virtual env containing tensorboard
tensorboard --logdir logs
```

- The tensorboard can be viewed on the local machine at:

```bash
http://127.0.0.1:16006
```
