import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt

def onehot_encode(label, device, n_class):
    """
    カテゴリカル変数のラベルをOne-Hoe形式に変換する
    :param label: 変換対象のラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :param n_class: ラベルのクラス数
    :return:
    """
    eye = torch.eye(n_class, device=device)
    # ランダムベクトルあるいは画像と連結するために(B, c_class, 1, 1)のTensorにして戻す
    return eye[label].view(-1, n_class, 1, 1)

def concat_image_label(image, label, device, n_class):
    """
    画像とラベルを連結する
    :param image: 画像
    :param label: ラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :param n_class: ラベルのクラス数
    :return: 画像とラベルをチャネル方向に連結したTensor
    """
    B, C, H, W = image.shape    # 画像Tensorの大きさを取得
    
    oh_label = onehot_encode(label, device, n_class)         # ラベルをOne-Hotベクトル化
    oh_label = oh_label.expand(B, n_class, H, W)    # 画像のサイズに合わせるようラベルを拡張する
    return torch.cat((image, oh_label), dim=1)      # 画像とラベルをチャネル方向（dim=1）で連結する


def concat_noise_label(noise, label, device, n_class):
    """
    ノイズ（ランダムベクトル）とラベルを連結する
    :param noise: ノイズ
    :param label: ラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :return: ノイズとラベルを連結したTensor
    """
    oh_label = onehot_encode(label, device, n_class)     # ラベルをOne-Hotベクトル化
    return torch.cat((noise, oh_label), dim=1)  # ノイズとラベルをチャネル方向（dim=1）で連結する



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
    plt.xlabel('iter')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.plot(range(1, 1 + len(Dx)), Dx, label="realimg")
    plt.plot(range(1, 1 + len(DGbefore)), DGbefore, label="generateimg_before")
    plt.plot(range(1, 1 + len(DGafter)), DGafter, label="generateimg_after")
    plt.xlabel('iter')
    plt.ylabel('identification-signal')
    plt.legend()

    plt.show()



def train_gan(epochs, dl, device, nz, netD, netG, criterion, optimG, optimD, \
    display_interval, \
        testgen_interval, test_fixed_noise, test_genimg_dir):
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

            if epoch == 0 and itr == 0:     # 初回に本物画像を保存する
                utils.save_image(real_image, '{}/real_samples.png'.format(test_genimg_dir),
                                  normalize=True, nrow=10)

            # ログ出力用データの保存
            D_losses.append(errD.item())
            G_losses.append(errG.item())
            D_x_out.append(D_x)
            D_G_z1_out.append(D_G_z1)
            D_G_z2_out.append(D_G_z2)


        #固定ノイズでテスト、生成された画像をディレクトリに保存
        if (epoch + 1) % testgen_interval == 0:
            fake_image = netG(test_fixed_noise)
            utils.save_image(fake_image.detach(), \
                '{}/fake_samples_epoch_{:03d}.png'.format(test_genimg_dir, epoch + 1),
                        normalize=True, nrow=10)
    

    return D_losses, G_losses, D_x_out, D_G_z1_out, D_G_z2_out


def train_cgan(epochs, dl, classes, device, nz, netD, netG, criterion, optimG, optimD, \
    display_interval, \
        testgen_interval, test_fixed_noise, test_genimg_dir):
    G_losses = []
    D_losses = []
    D_x_out = []
    D_G_z1_out = []
    D_G_z2_out = []

    # 学習のループ
    for epoch in range(epochs):
        for itr, data in enumerate(dl):
            real_image = data[0].to(device)     # 本物画像
            real_label = data[1].to(device)     # 本物画像に対応するラベル
            # 本物画像とラベルを連結
            real_image_label = concat_image_label(real_image, real_label, device, classes) 
            sample_size = real_image.size(0)    # 画像枚数

            # 標準正規分布からノイズを生成
            noise = torch.randn(sample_size, nz, 1, 1, device=device)
            # 生成画像生成用のラベル
            fake_label = torch.randint(classes, (sample_size,), dtype=torch.long, device=device)
            # ノイズとラベルを連結
            fake_noise_label = concat_noise_label(noise, fake_label, device, classes)        
            # 本物画像に対する識別信号の目標値「1」
            real_target = torch.full((sample_size,), 1., device=device)
            # 生成画像に対する識別信号の目標値「0」
            fake_target = torch.full((sample_size,), 0., device=device)
            
            ############################
            # 識別器Dの更新
            ###########################
            netD.zero_grad()    # 勾配の初期化
            
            # 識別器Dで本物画像とラベルの組み合わせに対する識別信号を出力
            output = netD(real_image_label)
            # 本物画像に対する識別信号の損失値
            errD_real = criterion(output, real_target)

            D_x = output.mean().item()  # 本物画像の識別信号の平均

            fake_image = netG(fake_noise_label)  # 生成器Gでラベルに対応した生成画像を生成
            # 生成画像とラベルを連結
            fake_image_label = concat_image_label(fake_image, fake_label, device, classes)   
            
            # 識別器Dで本物画像に対する識別信号を出力
            output = netD(fake_image_label.detach()) 
            # 生成画像に対する識別信号の損失値
            errD_fake = criterion(output, fake_target)  
            D_G_z1 = output.mean().item()# 生成画像の識別信号の平均

            errD = errD_real + errD_fake    # 識別器Dの全体の損失
            errD.backward()    # 誤差逆伝播
            optimD.step()   # Dのパラメーターを更新

            ############################
            # 生成器Gの更新
            ###########################
            netG.zero_grad()    # 勾配の初期化
            
            output = netD(fake_image_label)     # 更新した識別器Dで改めて生成画像とラベルの組み合わせに対する識別信号を出力
            errG = criterion(output, real_target)   # 生成器Gの損失値。Dに生成画像を本物画像と誤認させたいため目標値は「1」
            errG.backward()     # 誤差逆伝播
            D_G_z2 = output.mean().item()# 更新した識別器Dによる生成画像の識別信号の平均

            optimG.step()   # Gのパラメータを更新

            if itr % display_interval == 0: 
                print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                    .format(epoch + 1, epochs,
                            itr + 1, len(dl),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # ログ出力用データの保存
            D_losses.append(errD.item())
            G_losses.append(errG.item())
            D_x_out.append(D_x)
            D_G_z1_out.append(D_G_z1)
            D_G_z2_out.append(D_G_z2)
        
        
        #固定ノイズでテスト、生成された画像をディレクトリに保存
        if (epoch + 1) % testgen_interval == 0:
            fake_image = netG(test_fixed_noise)
            utils.save_image(fake_image.detach(), \
                '{}/fake_samples_epoch_{:03d}.png'.format(test_genimg_dir, epoch + 1),
                        normalize=True, nrow=10)

    return D_losses, G_losses, D_x_out, D_G_z1_out, D_G_z2_out
           