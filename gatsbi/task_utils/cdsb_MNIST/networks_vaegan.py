import torch
from torch import nn
#import argparse
#from  glm_utils import tuple_type

    
class VAEGAN(nn.Module):
    
    def __init__(self,
                 
                 # encoder params
                 #### f_g; can be 64/128
                 filters_gen=64,
                 residual_layers=3,
                 conv_size = (3,3),
                 norm = None,             
                            
                 
                 filters_disc=64,

                 #n; can be increased
                 noise_channels=50,
                 
                 latent_variables=50,
                 forceconv=False
                 ) -> None:
        super().__init__()

        
        self.grouped_conv2d_reduce_time = torch.nn.Conv2d( 6*4, 6, (1,1), groups=6, bias=False )
        
        self.encoder = VAEGAN_encoder( filters_gen,
                                        residual_layers,
                                        conv_size,
                                        norm=norm,
                                        latent_variables=latent_variables,
                                        forceconv=forceconv)
        
        self.decoder = VAEGAN_decoder(
                  filters_gen,
                  norm=norm,
                  latent_variables=latent_variables,
                    forceconv=forceconv)
        
        
                                
        self.latent_variables = latent_variables
        self.noise_channels = noise_channels
        
    def forward(self, variable_fields): #ensemble_size=1
        
        image = self.generator(variable_fields)
        
        return image
    
    def generator(self, variable_fields): #ensemble_size=1

        # _, _, _, d = variable_fields.shape

        # # grouped convolution to reduce time dimension in variable fields
        # variable_fields = einops.rearrange( variable_fields, '(b t) d h w -> b (t d) h w', t=4)
        # variable_fields = self.grouped_conv2d_reduce_time( variable_fields ) #(b, d, h, w )
                
                            
        z_mean, z_logvar = self.encoder(variable_fields)      

        noise = torch.randn( (variable_fields.shape[0],
                                self.noise_channels,
                                *variable_fields.shape[2:]),
                                device=z_logvar.device)
                    
        image = self.decoder(z_mean, z_logvar, noise )
        return image
                    
    # def parse_model_args(parent_parser):
    #     model_parser = argparse.ArgumentParser(
    #         parents=[parent_parser], add_help=True, allow_abbrev=False)
        
    #     model_parser.add_argument("--filters_gen", default=84,type=int)
    #     model_parser.add_argument("--residual_layers", default=3, type=int)
    #     model_parser.add_argument("--conv_size", default=(3,3), type=tuple_type)
    #     model_parser.add_argument("--latent_variables", default=32, type=int)
    #     model_parser.add_argument("--noise_channels", default=32, type=int)
    #     model_parser.add_argument("--forceconv", action='store_true', default=False )
        
    #     # model_parser.add_argument("--latent_variables", default=32, type=int)
        
    #     model_parser.add_argument("--filters_disc", default=None,type=int)
    #     model_parser.add_argument("--norm", default=None,type=str)
    #     model_args = model_parser.parse_known_args()[0]
        
    #     if model_args.filters_disc is None: 
    #         model_args.filters_disc = model_args.filters_gen * 4
        
    #     return model_args




class VAEGAN_encoder(nn.Module):
    
    def __init__(self,
                    filters_gen=128,
                    residual_layers=3,
                    # input_channels=9,
                    # img_shape=(100, 100),
                    conv_size=(3, 3),
                    # padding_mode='reflect',
                    padding_mode='zeros',
                    stride=1,
                    relu_alpha=0.2,
                    norm=None,
                    dropout_rate=None,
                    latent_variables=50,
                    forceconv=True
                    ) -> None:
        super().__init__()
                
        # num. of pictures passed as input
        var_field_inp = 1

        self.residual_block = nn.Sequential(
            *[ ResidualBlock(
                    filters_gen if idx!=0 else var_field_inp, filters_gen, conv_size=(3,3),
                    stride=stride, relu_alpha=relu_alpha, norm=norm,
                    dropout_rate=dropout_rate, padding_mode=padding_mode,
                    force_1d_conv=forceconv) 
                for idx in range(residual_layers)
                ]
        )
        
        self.conv2d_mean = nn.Sequential( 
                                nn.Conv2d( filters_gen, latent_variables , kernel_size=1, padding='valid'),
                                nn.LeakyReLU(relu_alpha))
        
        self.conv2d_logvars = nn.Sequential( 
                                nn.Conv2d( filters_gen, latent_variables , kernel_size=1, padding='valid'),
                                nn.LeakyReLU(relu_alpha)
                                )
            
    def forward(self, variable_fields):
        """_summary_
        Args:
            variable_fields (_type_): (b, c, h, w)
        """
        x = variable_fields
        
        # Residual Block
        x1 = self.residual_block(x)
        
        # Add noise to log_vars
        z_mean = self.conv2d_mean(x1)
        z_logvar = self.conv2d_logvars(x1)
                    
        return z_mean, z_logvar




class VAEGAN_decoder(nn.Module):
    def __init__(self,
                 filters_gen,
                 relu_alpha=0.2, 
                 stride=1,
                 norm=None,
                 dropout_rate=0.0,
                #  padding_mode='reflect',
                 padding_mode='zeros',
                 conv_size=(3,3),
                 num_layers_res1=3,
                 num_layers_res2=3,
                 forceconv=True,
                 latent_variables=50
                 ):
        super().__init__()
        
        # if arch == "forceconv-long"
        self.residual_block1 = nn.Sequential(
            *( ResidualBlock(
                    filters_gen if idx!=0 else latent_variables , filters_gen, conv_size=conv_size,
                    stride=stride, relu_alpha=relu_alpha, norm=norm,
                    dropout_rate=dropout_rate, padding_mode=padding_mode,
                    force_1d_conv=forceconv) 
                for idx in range(num_layers_res1) )
        )
        
        # Upsampling from (10,10) to (100,100) with alternating residual blocks
        us_block_channels = [2*filters_gen, filters_gen]
        
        const_field_inp = 1
        const_field_k1 = (1.0, 1.0 )
        const_field_k2 = (4.0, 4.0 )
        self.upsample_residual_block = nn.Sequential(
            nn.UpsamplingBilinear2d( scale_factor = const_field_k1),
            ResidualBlock(filters_gen, us_block_channels[0], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv) ,
            nn.UpsamplingBilinear2d( scale_factor =  const_field_k2),
            # ResidualBlock(filters_gen, us_block_channels[1], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv)            
            ResidualBlock(us_block_channels[0], us_block_channels[1], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv)            
        )
                
        self.residual_block2 = nn.Sequential(
                                    *( ResidualBlock(
                                            us_block_channels[-1] if idx==0 else filters_gen, 
                                            filters_gen, conv_size=(3,3),
                                            stride=stride, relu_alpha=relu_alpha, norm=None,
                                            dropout_rate=dropout_rate, padding_mode=padding_mode,
                                            force_1d_conv=forceconv) 
                                        for idx in range(num_layers_res2) )
                                    )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters_gen, 1, (1,1)),
            nn.Softplus()
            # nn.ReLU()
        )
    
    def forward(self, z_mean , z_logvar, noise):
        """_summary_
            Args:
                x (c1, h1, w1): _description_
        """
        
        # Generated noised output
        sample_z = torch.mul( noise, torch.exp(z_logvar*0.5) ) + z_mean
        
        # residual block 1
        x = self.residual_block1(sample_z)
        #print("SHAPE X: ", x.shape)
        # upsampling residual block
        x1 = self.upsample_residual_block(x)
        #print("SHAPE X1: ", x1.shape)
        # residual block 2
        x2 = self.residual_block2(x1)
        #print("SHAPE X2: ", x2.shape)
        # softplus        
        x3 = self.output_layer(x2)    
        #print("SHAPE X3: ", x3.shape)
        return x3
        
##########################################################################
#                Residual Block (2x inner block + highway)
##########################################################################

class ResidualBlock_innerblock(nn.Module):
    
    def __init__(self, in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate) -> None:
        super().__init__()
        
        self.act = nn.LeakyReLU(relu_alpha)
        
        self.conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=conv_size,
                padding='same',
                padding_mode=padding_mode #convert all padding tf words to equivalent pytorch ones
            )
        torch.nn.init.xavier_uniform_(self.conv2d.weight)

        self.bn = nn.BatchNorm2d(filters) if norm=="batch" else nn.Identity()           
        
        self.do = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
                

    def forward(self, x):
        
        x = self.act(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.do(x)        
        
        return x



class ResidualBlock_highway(nn.Module):
    
    def __init__(self, filters, in_channels, force_1d_conv, stride) -> None:
        super().__init__()
        
        
        self.ap2d =  nn.AvgPool2d( kernel_size=(stride,stride))  if stride > 1 else nn.Identity()
        
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                            out_channels=filters,
                                            kernel_size=(1, 1))  if (filters != in_channels) \
                        or force_1d_conv else nn.Identity()
                        
        # if len(highway_block)==0:
        #     return torch.nn.Identity()
    
    def forward(self, x):
        x = self.ap2d(x)
        x = self.conv2d(x)
        return x
    
    

class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 filters, conv_size=(3, 3), stride=1, 
                 relu_alpha=0.2, norm=None, dropout_rate=None, 
                 padding_mode='reflect', force_1d_conv=False
                 ) -> None:
        super().__init__()
        
        # in_channels = x.shape[-3]
        
        #inner highway_block
        # self.highway_block = residualblock_highway(filters, in_channels, force_1d_conv, stride)
        self.highway_block = ResidualBlock_highway(filters, in_channels, force_1d_conv, stride)
        
        # first block of activation and 3x3 convolution
        self.block1 = ResidualBlock_innerblock(in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        # self.block1 = residualblock_innerblock(in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        
        # second block of activation and 3x3 convolution
        self.block2 = ResidualBlock_innerblock(filters, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        # self.block2 = residualblock_innerblock(filters, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
                 
    def forward(self, x):
        """_summary_
        Args:
            x (_type_): (b, c, h, w)
        """
        
                        
        x1 = self.block1(x)
        
        x2 = self.block2(x1)
        
        x3 = x2 + self.highway_block(x)
        
        return x3
        

