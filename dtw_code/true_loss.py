from tslearn.metrics import dtw, dtw_path


class TrueLoss:
    def __init__(self, config):
        self.config = config
    
    def compute(self, y, y_hat):
        loss_dtw = 0
        for k in range(self.config.batch_size):
            target_k_cpu = y[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = y_hat[k,:,0:1].view(-1).detach().cpu().numpy()
            path, sim = dtw_path(target_k_cpu, output_k_cpu)
            loss_dtw += sim
        return loss_dtw