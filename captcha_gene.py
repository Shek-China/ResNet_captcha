"""
随机生成颜色
gene_color()
随机生成噪点
gene_noise(draw, width, height)
随机生成干扰线
gene_line(draw, width, height)
用于创建简单的验证码
get_text(mode='number',count=4)
获得随机的 字母/数字 任意长度的字符串
get_random_font()
返回一个随机的字体
get_captcha(dirname='./', mode='mix', count=4, random_font=False)
在指定文件夹生成验证码图片，并且返回文件名称
create_pic(num, dirname='./')
在指定文件夹、生成指定数量的图片
"""
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np

# 背景颜色(默认）
bgcolor = (255, 255, 255)
# 字体颜色
fontcolor = (0, 0, 0)
# 干扰线颜色
# linecolor = random.sample(COLOR_LIST,3)
# 是否绘制干扰线
draw_line = True
# 加入干扰线的上下限
line_number = tuple(np.arange(1, 6))
# 生成验证码大小
FONT_SIZE = 60
# 验证码字符串长度
TOTAL_LENGTH = 5
# 颜色表
COLOR_LIST = list(np.arange(256))
# 噪点数
NOISE_COUNT = 20
# 是否添加噪点
draw_noise = True


# 不同元素的颜色生成
def gene_color():
    color = tuple(random.sample(COLOR_LIST, 3))
    return color


# 三种模式获得一个随机的指定位数的字符串: number-纯数字, word-纯字母, mix-数字和字母混合
def get_text(mode='number', count=TOTAL_LENGTH - 1):
    number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    word_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'Z', 'X', 'Y']
    if mode == 'number':
        return ''.join(random.sample(number_list, count))
    elif mode == 'word':
        return ''.join(random.sample(word_list, count))
    else:
        return ''.join(random.sample(number_list + word_list, count))


# 获得一个随机的字体文件路径
def get_random_font():
    # 获取字体目录下所有文件
    font_path = 'C:/Windows/Fonts/'
    font_list = os.listdir(font_path)

    # 遍历所有文件，保留ttf文件，返回这个列表
    ttf_list = []
    for font_file in font_list:
        if font_file.endswith('.ttf'):
            ttf_list.append(font_file)

    my_font = random.choice(ttf_list)
    return font_path + my_font


# 字符串写入图片，生成验证码图片
def get_captcha(dirname='./', mode='mix', count=4, random_font=False):
    # 导入字体
    if not random_font:
        font_path = 'C:/Windows/Fonts/arial.TTF'
    else:
        font_path = get_random_font()
        font = ImageFont.truetype(font_path, FONT_SIZE)

    # 生成字符串
    text = get_text(mode, count)

    # 根据指定字体字符串的尺寸，计算出字符串放在图片中间时的坐标
    font_width, font_height = font.getsize(text)
    # size=(宽度,高度) 图片大小会根据字数调整，同时字符串总数会在图片的中间位置
    # width = font_width + 20 + count
    # height = font_height + 20 + count // 5
    width = 240
    height = 120
    size = width, height
    start_point = ((width - font_width) / 2, (height - font_height) / 2)

    # 创建图片画布
    image = Image.new('RGBA', size, bgcolor)

    # 创建画笔并使用字符串填充图片AAAA
    draw = ImageDraw.Draw(image)
    draw.text(start_point, text, font=font, fill=gene_color())

    if draw_line:
        gene_line(draw, width, height)
    if draw_noise:
        gene_noise(draw, width, height)
    # (1, -0.1, 0, -0.1, 1, 0)是三组参数构成的参数矩阵，用于将图像进行扭曲，配合AFFINE使用
    image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.1, 0, -0.1, 1, 0), Image.BILINEAR)  # 创建扭曲
    image = image.filter(ImageFilter.BLUR)  # 添加模糊滤镜
    image = image.filter(ImageFilter.EDGE_ENHANCE)  # 滤镜，边界加强
    # 保存图片到当前文件夹
    file_path = os.path.join(dirname, text + '.png')
    image.save(file_path)
    return file_path


# 绘制干扰线
def gene_line(draw, width, height):
    times = random.randint(1, 5)
    for i in range(times):
        linecolor = gene_color()
        begin = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([begin, end], fill=linecolor, width=3)


# 绘制噪点
def gene_noise(draw, width, height):
    for i in range(NOISE_COUNT):
        x = random.randint(0, width)
        y = random.randint(0, height)
        noise_color = gene_color()
        draw.point((x, y), fill=noise_color)


def create_pic(num, dirname='./', mode='mix', random_font=False):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for i in range(num):
        file_path = get_captcha(dirname, mode, TOTAL_LENGTH, random_font)
        print('创建 %s 成功...' % file_path)


if __name__ == '__main__':
    for i in range(200):
        create_pic(5, 'test', 'mix', True)
