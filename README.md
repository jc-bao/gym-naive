# Naive gym environment for robot learning

## Environments

### `NaivePickAndPlace-v0`

#### Configration

```python
config = {
  'dim': 2, # dimension of environment, 2 or 3
  '''
  Reward type
  [Dense]
    if not self.if_grasp:
      reward = 0.5 * (1 - np.tanh(2.0 * self.d_a2o))
    else:
      reward = (0.5 + 0.5*(1 - np.tanh(1.0 * self.d_o2g)))
  [dense_o2g]
    reward = -self.d_o2g
  [sparse]
    reward = (self.d_o2g<self.error).astype(np.float32)
  [dense_diff_o2g]
    reward = self.d_old - self.d_o2g
  '''
  'reward_type': 'dense', # reward type
  'error': 0.05, 
  'use_grasp': True, 
  'vel': 0.5, 
  'init_grasp_rate': 0.0,
  'use_her': False, 
  'mode': 'static', # if the object can move together
}
env = gym.make('NaivePickAndPlace-v0', config = config)
```

#### Demo

```python
python gym_naive/naive_pac.py 
```

| Reward_type = dense. Dim = 2 Vel=0.2                         | Reward_type = dense_o2g. Dim = 2 Vel=0.2                     | Reward_type = dense_diff_o2g. Dim = 2 Vel=0.2                | Reward_type = sparse. Dim = 2 Vel=0.2                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20211031095236932](https://tva1.sinaimg.cn/large/008i3skNly1gvy99fwyuaj30f80bxwek.jpg) | ![image-20211031095320695](https://tva1.sinaimg.cn/large/008i3skNly1gvy9a6x8wjj30fp0bqwei.jpg) | ![image-20211031095524008](https://tva1.sinaimg.cn/large/008i3skNly1gvy9cc7hbkj30gd0bxaa4.jpg) | ![image-20211031100747346](https://tva1.sinaimg.cn/large/008i3skNly1gvy9p893m1j30g50bs74c.jpg) |
| ![image-20211031095247064](https://tva1.sinaimg.cn/large/008i3skNly1gvy99lrfmej30fp0c274i.jpg) | ![image-20211031095441871](https://tva1.sinaimg.cn/large/008i3skNly1gvy9blisxlj30g80cm0sz.jpg) | ![image-20211031100655839](https://tva1.sinaimg.cn/large/008i3skNly1gvy9obyc5xj30g30co74k.jpg) | ![image-20211031100756213](https://tva1.sinaimg.cn/large/008i3skNly1gvy9pdcgrqj30fq0ckq32.jpg) |

