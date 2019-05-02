import torch

class ToTensor(object):
    def __call__(self, states, device):
        ranges_batch = []
        target_points_batch = []
        for state in states:
            ranges = state['ranges']
            target_point = state['target_point']
            ranges = torch.tensor(ranges, device=device, dtype=torch.float32, requires_grad=False)
            target_point = torch.tensor(target_point, device=device, dtype=torch.float32, requires_grad=False)
            ranges_batch.append(ranges)
            target_points_batch.append(target_point)
        ranges_batch = torch.stack(tuple(ranges_batch))
        target_points_batch = torch.stack(tuple(target_points_batch))
        state_batch = ranges_batch, target_points_batch
        return state_batch


class ToRecurrentStatesTensor(ToTensor):
    def __call__(self, batch, device):
        ranges_batch = []
        target_points_batch = []
        for seq in batch:
            ranges_seq, target_points_seq = super(ToRecurrentStatesTensor, self).__call__(seq, device)
            ranges_batch.append(ranges_seq)
            target_points_batch.append(target_points_seq)
        ranges_batch = torch.stack(tuple(ranges_batch)).permute(1, 0, 2, 3)
        target_points_batch = torch.stack(tuple(target_points_batch)).permute(1, 0, 2)
        return ranges_batch, target_points_batch
