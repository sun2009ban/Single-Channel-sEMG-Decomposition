import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import visdom
from torch.autograd import grad as torch_grad
import pdb

def _gradient_penalty(gp_weight, real_data, generated_data, D_net, device):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = interpolated.detach()
    interpolated = torch.tensor(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = D_net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()

def train_dc_gan(train_data, D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, filter=None, show_every=10, 
                noise_size=100, gaussian_noise=True, num_epochs=20, d_every=1, g_every=1, save_every=100, add_noise=False, noise_amp=0.01, gradient_penalty=10,
                use_memory=False, memory=None, gradient_clip=False, device=torch.device('cpu'), dtype=torch.float, save_dir='./saveTorch', 
                save_output_dir='./saveOutput', use_visdom=False):
    
    if use_visdom:
        vis = visdom.Visdom()

    g_loss = []
    d_loss = []
    g_grad_norm = []
    d_grad_norm = []

    for epoch in range(num_epochs):
        #D_scheduler.step()
        #G_scheduler.step()
        
        for ii, x in enumerate(train_data):
            bs = x.shape[0]
            x = x.view(bs, 1, -1)
            
            if add_noise:
                x = x + noise_amp * torch.randn(x.size())
            # 判别网络          
            if ii % d_every == 0:
                
                D_optimizer.zero_grad()

                if gaussian_noise:
                    sample_noise = torch.randn(bs, noise_size, 1) # 均值为0,方差为1的正态分布
                else:
                    sample_noise = torch.empty(bs, noise_size, 1).uniform_(-1, 1) # 在(-1,1)之间的均匀分布

                ''' 全是正确样本的batch '''
                real_data = x.to(device, dtype)
                if filter is not None:
                    real_data = torch.matmul(real_data, filter)
                logits_real = D_net(real_data) # 判别网络得分             
                
                ''' 全是错误样本的batch '''
                
                if use_memory and np.random.rand() <= 0.2 and len(memory) > bs:
                    fake_images = memory.sample(bs)
                    fake_images = torch.cat(fake_images, 0)
                    fake_images = fake_images.unsqueeze(1)
                    fake_images = fake_images.to(device, dtype)
                else:
                    g_fake_seed = sample_noise.to(device, dtype)
                    fake_images = G_net(g_fake_seed).detach() # 生成的假的数据, detach()使得fake_images作为叶子节点，不会更新G_net的参数t
                    if filter is not None:
                        fake_images = torch.matmul(fake_images, filter)
                logits_fake = D_net(fake_images) # 判别网络得分

                errD = discriminator_loss(logits_real, logits_fake)

                errD_gp = _gradient_penalty(gradient_penalty, real_data, fake_images, D_net, device) # 0.1的时候刚好平衡

                errD_gp.backward(retain_graph=True) # 加入gradien penalty
                errD.backward()
                
                if gradient_clip:
                    torch.nn.utils.clip_grad.clip_grad_norm_(D_net.parameters(), 0.5) # gradient clip

                '''把错误的样本存起来'''
                if use_memory:
                    for index in range(fake_images.size()[0]):
                        memory.push(fake_images[index].data)
                
                D_optimizer.step() # 优化判别网络
                #G_optimizer.ascent_half_step()
            
            # 生成网络
            if ii % g_every == 0:

                G_optimizer.zero_grad()

                ''' 全是正确样本的batch '''
                real_data = x.to(device, dtype)
                if filter is not None:
                    real_data = torch.matmul(real_data, filter)
                logits_real = D_net(real_data) # 判别网络得分     

                if gaussian_noise:
                    sample_noise = torch.randn(bs, noise_size, 1) # 均值为0,方差为1的正态分布
                else:
                    sample_noise = torch.empty(bs, noise_size, 1).uniform_(-1, 1) # 在(-1,1)之间的均匀分布

                g_fake_seed = sample_noise.to(device, dtype)
                fake_images = G_net(g_fake_seed) # 生成的假的数据
                if filter is not None:
                    fake_images = torch.matmul(fake_images, filter)

                gen_logits_fake = D_net(fake_images)
                errG = generator_loss(logits_real, gen_logits_fake) # 生成网络的 loss
                
                errG.backward()
                
                if gradient_clip:  
                    torch.nn.utils.clip_grad.clip_grad_norm(G_net.parameters(), 5) # gradient clip

                G_optimizer.step() # 优化生成网络
                #D_optimizer.ascent_half_step()

            if ii % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(ii + 1, errD.item(), errG.item()))
                
                # 记录loss
                d_loss.append(errD.item()) 
                g_loss.append(errG.item())
                
                d_loss_tensor = torch.Tensor(d_loss)
                g_loss_tensor = torch.Tensor(g_loss)
                
                '''
                显示loss
                '''
                '''
                if use_visdom:
                    vis.line(X=np.arange(len(g_loss_tensor)), Y=g_loss_tensor, win='loss', name='G_loss')
                    vis.line(X=np.arange(len(d_loss_tensor)), Y=d_loss_tensor, win='loss', name='D_loss', update='append')
                '''

                if use_visdom:
                    # 计算gradient的nrom，看看梯度对不对
                    nb_params_g = 0
                    g_grad_norm_sum = 0
                    for param in G_net.parameters():
                        if param.grad is not None:
                            g_grad_norm_sum += param.grad.norm().item()
                            nb_params_g += 1
                    g_grad_norm_sum /= nb_params_g

                    nb_params_d = 0
                    d_grad_norm_sum = 0
                    for param in D_net.parameters():
                        if param.grad is not None:
                            d_grad_norm_sum += param.grad.norm().item()
                            nb_params_d += 1
                    d_grad_norm_sum /= nb_params_d

                    g_grad_norm.append(g_grad_norm_sum)
                    d_grad_norm.append(d_grad_norm_sum)

                    '''
                    显示gradient norm的代码
                    '''
                    vis.line(X=np.arange(len(g_grad_norm)), Y=g_grad_norm, win='grad', name='G_grad')
                    vis.line(X=np.arange(len(d_grad_norm)), Y=d_grad_norm, win='grad', name='D_grad', update='append')


                # 输出真实数据和生成数据
                plot_fake = fake_images.data.cpu().numpy()[:64]
                plot_real = real_data.data.cpu().numpy()[:64]

                plot_fake = np.reshape(plot_fake, (plot_fake.shape[0], -1))
                plot_real = np.reshape(plot_real, (plot_real.shape[0], -1))           
                
                '''
                显示real和fake的数据
                '''
                '''
                if use_visdom:
                    vis.line(X=np.arange(len(plot_fake[0])), Y=plot_fake[0], win='fake')
                    vis.line(X=np.arange(len(plot_real[0])), Y=plot_real[0], win='real')
                    vis.images(plot_fake, win='fake') # 生成网络生成的数据
                    vis.images(plot_real, win='real') # 真实数据
                '''

                # 以numpy形式存储real和fake
                maximum_plot_size = 200
                plt.figure()
                plt.plot(np.arange(len(g_loss_tensor)), g_loss_tensor.data.numpy())
                plt.plot(np.arange(len(d_loss_tensor)), d_loss_tensor.data.numpy())
                plt.savefig(os.path.join(save_output_dir, 'loss_' + str(epoch % maximum_plot_size) + '_' + str(ii) + '.png'))
                plt.close()

                plt.figure()
                plt.plot(np.arange(len(plot_fake[0])), plot_fake[0])
                plt.savefig(os.path.join(save_output_dir, 'fake_' + str(epoch % maximum_plot_size) + '_' + str(ii) + '.png'))
                plt.close()

                plt.figure()
                plt.plot(np.arange(len(plot_real[0])), plot_real[0])
                plt.savefig(os.path.join(save_output_dir, 'real_' + str(epoch % maximum_plot_size) + '_' + str(ii) + '.png'))   
                plt.close()
                
            if ii % save_every == 0:             
                maximum_save_size = 100 # 只保留最近1000个结果
                # 存储模型
                torch.save(D_net.state_dict(), os.path.join(save_dir, 'D_DC_' + str(epoch % maximum_save_size) + '_' + str(ii) + '.pth'))
                torch.save(G_net.state_dict(), os.path.join(save_dir, 'G_DC_' + str(epoch % maximum_save_size) + '_' + str(ii) + '.pth'))   
                