import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt

def display_GAN_curv(fig_w, fig_h, lblfs, sclfs, Dloss, Gloss, Dx, DGbefore, DGafter):
    rcparams_dic = {
        'figure.figsize': (fig_w,fig_h),
        'axes.labelsize': lblfs,
        'xtick.labelsize': sclfs,
        'ytick.labelsize': sclfs,
    }
    plt.rcParams.update(rcparams_dic)

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 1 + len(Dloss)), Dloss, label="discriminator")
    plt.plot(range(1, 1 + len(Gloss)), Gloss, label="generator")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.plot(range(1, 1 + len(Dx)), Dx, label="realimg")
    plt.plot(range(1, 1 + len(DGbefore)), DGbefore, label="generateimg_before")
    plt.plot(range(1, 1 + len(DGafter)), DGafter, label="generateimg_after")
    plt.xlabel('Epochs')
    plt.ylabel('identification-signal')
    plt.legend()

    plt.show()



def train_gan(epochs, dl, device, nz, netD, netG, criterion, optimG, optimD, \
    display_interval):
    G_losses = []
    D_losses = []
    D_x_out = []
    D_G_z1_out = []
    D_G_z2_out = []


    for epoch in range(epochs):
        for itr, data in enumerate(dl):
            real_image = data[0].to(device)     # 本物画像
            sample_size = real_image.size(0)    # 画像枚数

            # 標準正規分布からノイズを生成
            noise = torch.randn(sample_size, nz, 1, 1, device=device)
            # 本物画像に対する識別信号の目標値「1」
            real_target = torch.full((sample_size,), 1., device=device)
            # 生成画像に対する識別信号の目標値「0」
            fake_target = torch.full((sample_size,), 0., device=device) 

            ############################
            # 識別器Dの更新
            ###########################
            netD.zero_grad()    # 勾配の初期化

            output = netD(real_image)   # 識別器Dで本物画像に対する識別信号を出力
            errD_real = criterion(output, real_target)  # 本物画像に対する識別信号の損失値
            D_x = output.mean().item()  # 本物画像の識別信号の平均

            fake_image = netG(noise)    # 生成器Gでノイズから生成画像を生成

            output = netD(fake_image.detach())  # 識別器Dで本物画像に対する識別信号を出力
            errD_fake = criterion(output, fake_target)  # 生成画像に対する識別信号の損失値
            D_G_z1 = output.mean().item()  # 生成画像の識別信号の平均

            errD = errD_real + errD_fake    # 識別器Dの全体の損失
            errD.backward()    # 誤差逆伝播
            optimD.step()   # Dのパラメーターを更新

            ############################
            # 生成器Gの更新
            ###########################
            netG.zero_grad()    # 勾配の初期化

            output = netD(fake_image)   # 更新した識別器Dで改めて生成画像に対する識別信号を出力
            errG = criterion(output, real_target)   # 生成器Gの損失値。Dに生成画像を本物画像と誤認させたいため目標値は「1」
            errG.backward()     # 誤差逆伝播
            D_G_z2 = output.mean().item()  # 更新した識別器Dによる生成画像の識別信号の平均

            optimG.step()   # Gのパラメータを更新

            if itr % display_interval == 0: 
                print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                      .format(epoch + 1, epochs,
                              itr + 1, len(dl),
                              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # if epoch == 0 and itr == 0:     # 初回に本物画像を保存する
            #     utils.save_image(real_image, '{}/real_samples.png'.format(outf),
            #                       normalize=True, nrow=10)

            # ログ出力用データの保存
            D_losses.append(errD.item())
            G_losses.append(errG.item())
            D_x_out.append(D_x)
            D_G_z1_out.append(D_G_z1)
            D_G_z2_out.append(D_G_z2)
    

    return D_losses, G_losses, D_x_out, D_G_z1_out, D_G_z2_out
