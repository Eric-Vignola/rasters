from PIL import Image, ImageDraw, ImageFont
import numpy as np
import string, bz2, cPickle, os, glob

from itertools import groupby
from operator import itemgetter



# --- init font list --- #
FONTS = {}
RESOLUTION  = 256
LIB_PATH    = '%s/lib'%os.path.split(os.path.abspath(__file__).replace('\\','/'))[0]
RASTER_PATH = '%s/%s/'%(LIB_PATH, RESOLUTION)

for x in glob.glob('%s/*.r'%RASTER_PATH):
    x = x.replace('\\','/')
    name = os.path.splitext(os.path.split(x)[1])[0]
    FONTS[name.lower()] = x

DEFAULT_FONT = None
if FONTS:
    DEFAULT_FONT = FONTS[FONTS.keys()[0]]



def expand_path(path):
    path_ = None
    while path_ != path:
        path_ = path
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))).replace('\\','/')

    return path


def write_image(image_file, image):
    image_file = expand_path(image_file)
    
    folder = os.path.split(image_file)[0]
    if not os.path.isdir(folder):
        os.makedirs(folder)

    image.save(image_file)


def write_rasters(characters, raster_file, compress=True):
    raster_file = expand_path(raster_file)
    folder = os.path.split(raster_file)[0]

    if not os.path.isdir(folder):
        os.makedirs(folder)

    if compress:
        with bz2.BZ2File(raster_file, 'wb') as io:
            cPickle.dump(characters, io, cPickle.HIGHEST_PROTOCOL)

    else:
        with open(raster_file, 'wb') as io:
            cPickle.dump(characters, io, cPickle.HIGHEST_PROTOCOL)


def read_rasters(raster_file):
    raster_file = expand_path(raster_file)

    try:
        with bz2.BZ2File(raster_file, 'rb') as io:
            return cPickle.load(io)

    except:
        with open(raster_file, 'rb') as io:
            return cPickle.load(io)


def draw_text(txt, font):
    w, h = font.getsize(txt)
    canvas = Image.new('RGBA', (w, h))
    draw = ImageDraw.Draw(canvas)
    draw.text((0,0), txt, 'white', font)

    return canvas



def boundaries(image, horizontal=True):

    image = np.asarray(image)

    if horizontal:
        s = np.sum(image, axis=0)
    else:
        s = np.sum(image, axis=1)

    return np.where(s[:,0] > 0)[0]






def render_alphabet(font, sample_size=256, raster_file=None, proof_image=None, error_check=True):

    # init font and supported alphabet
    imagefont = ImageFont.truetype(font, sample_size)
    alphabet  = list(string.punctuation+string.ascii_letters+string.digits)
    alphabet[0], alphabet[1] = alphabet[1], alphabet[0]
    alphabet = ''.join(alphabet)


    #if proof_image:
        #w, h  = imagefont.getsize(alphabet)
        #image = Image.new('RGBA', (w, h*2))
        #draw  = ImageDraw.Draw(image)
        #draw.text((0,h/2), ' %s '%alphabet, 'white', font=imagefont)
        #write_image(proof_image, image)


    # render source characters (space out letters to help isolate)
    spacing   = 1
    spaces    = ' '*spacing
    alphabet_ = alphabet.replace('', spaces)#[spacing:-spacing]
    

    w, h  = imagefont.getsize(alphabet_)
    image = Image.new('RGBA', (w, h*2))
    draw  = ImageDraw.Draw(image)
    draw.text((0,h/2), alphabet_, 'white', font=imagefont)


    if proof_image:
        write_image(proof_image, image)


    # Extract letters from rendered alphabet by
    # collapsing each column and look for gaps.
    image    = np.array(image)
    boundary = boundaries(image, horizontal=False)
    image    = image[boundary]


    # find markers
    boundary = boundaries(image)
    markers  = []
    for k, g in groupby(enumerate(boundary), lambda (i, x): i-x):
        markers.append(np.array(map(itemgetter(1), g)))


    count_text = len(alphabet)
    count_markers = len(markers)
    delta = abs(count_text-count_markers)

    if delta > 0:
        if delta == 1:
            markers_ = [np.concatenate((markers[0],markers[1]))]
            markers  = markers_ + markers[2:]
        else:
            raise Exception('Failed to isolate letters: alphabet: %s --> markers: %s'%(count_text,count_markers))


    # create character dict with found pixels
    characters = {}
    for i,char in enumerate(alphabet):
        characters[char] = {'image':image[:,markers[i],:],
                            'width':markers[i].size}

    if error_check:
        keys  = characters.keys()
        count = len(keys)
        duplicates = set()

        for i in xrange(count):
            for ii in xrange(i,count-1):
                ii+=1

                test0 = keys[i]  in string.ascii_letters
                test1 = keys[ii] in string.ascii_letters

                if not (test0 and test1):
                    if characters[keys[i]]['image'].shape == characters[keys[ii]]['image'].shape:
                        if np.allclose(characters[keys[i]]['image'], characters[keys[ii]]['image']):
                            duplicates.add(keys[i])
                            duplicates.add(keys[ii])


        for d in duplicates:
            characters.pop(d,None)


    # Insert space
    width = np.array(draw_text(' ', font=imagefont))
    characters[' '] = {}
    characters[' ']['width'] = width.shape[1]
    characters[' ']['image'] = np.zeros(width.shape).astype('uint8')


    # estimate kerning for each character
    #https://stackoverflow.com/questions/2100696/can-i-get-the-kerning-value-of-two-characters-using-pil
    for char2 in characters:
        characters[char2]['kerning'] = {}

        for char1 in characters:
            w1 = characters[char1]['width']
            w2 = characters[char2]['width']
            w  = w1+w2

            if char1 == ' ' and char2 != ' ':
                kern = ' %s'%char2
                kern = draw_text(kern, font=imagefont)
                w = kern.width

            elif char1 != ' ' and char2 == ' ':
                kern0 = ' %s'%char1
                kern1 = ' %s '%char1
                kern0 = draw_text(kern0, font=imagefont)
                kern1 = draw_text(kern1, font=imagefont)
                w = (kern1.width - kern0.width) + w1

            else:
                kern = ' %s%s '%(char1,char2)
                kern = draw_text(kern, font=imagefont)
                kern = boundaries(kern)
                w = kern.size

            characters[char2]['kerning'][char1] = w - (w1 + w2)


    # insert minimum space
    min_width = draw_text(' .. ', font=imagefont)
    min_width = boundaries(min_width)

    markers  = []
    for k, g in groupby(enumerate(min_width), lambda (i, x): i-x):
        markers.append(np.array(map(itemgetter(1), g)))

    min_width = (markers[1][0] - markers[0][-1])+1
    min_width/=4
    characters[''] = {'width': min_width}


    # we're done! convert to image and save/return
    if error_check:
        for char in characters:
            if char:
                characters[char]['image'] = Image.fromarray(characters[char]['image']).convert('P')

        if len(characters) == 2:
            return None


    if raster_file:
        write_rasters(characters, raster_file)

    return characters



def composite(fg, bg, offset=(0,0), extend=True):
    ''' - alpha composites a foreground over a background
        - offset is relative to background
    '''
    width, height = bg.size
    fg_offset = list(offset)

    if extend:
        W,H = bg.size
        w,h = fg.size
        w+=abs(offset[0])
        h+=abs(offset[1])

        min_width,  max_width  = min(w,W), max(w,W)
        min_height, max_height = min(h,H), max(h,H)
        width  = (max_width  - min_width)  + min_width
        height = (max_height - min_height) + min_height

        bg_offset = [0,0]
        if fg_offset[0] < 0:
            bg_offset[0] = abs(fg_offset[0])
            fg_offset[0] = 0

        if fg_offset[1] < 0:
            bg_offset[1] = abs(fg_offset[1])
            fg_offset[1] = 0

        bg_ = Image.new('RGBA', (width, height))
        bg_.paste(bg, (bg_offset[0],bg_offset[1]))
        bg = bg_

    fg_ = Image.new('RGBA', (width, height))
    fg_.paste(fg, (fg_offset[0],fg_offset[1]))


    return Image.alpha_composite(bg,fg_)




def find_raster(font_name):

    if not isinstance(font_name, (str,unicode)):
        return font_name


    # Load raster file from built in list
    font_name = font_name.lower()

    if not font_name in FONTS:
        print 'LIST OF AVAILABLE FONTS:'
        print '------------------------'
        for x in sorted(FONTS.keys()):
            print x
        print '------------------------'

        raise Exception('Font unavailable: %s'%font_name)


    return read_rasters(FONTS[font_name])




def import_system_truetype(full_export=True, sample_size=256, border=10):

    import matplotlib.font_manager
    alphabet  = string.punctuation+string.ascii_letters+string.digits

    
    # Get a list of all fonts to export, ide them by name since there might be duplicates
    print 'Finding system fonts...'
    
    all_fonts_ = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    all_fonts  = {}
    
    for font_file in all_fonts_:
        font_name = os.path.splitext(os.path.split(font_file)[1])[0].lower()
        
        raster_file  = '%s/%s/%s.r'%(LIB_PATH, sample_size, font_name)
        preview_file = '%s/%s/previews/%s.png'%(LIB_PATH, sample_size, font_name)
        proof_file   = '%s/%s/previews/%s.proof.png'%(LIB_PATH, sample_size, font_name)
        
        go = True
        if not full_export and not os.path.isfile(raster_file):
            go = False
            
        if go:
            all_fonts[font_name] = {}
            all_fonts[font_name]['truetype'] = font_file.replace('\\','/')
            all_fonts[font_name]['raster']   = raster_file
            all_fonts[font_name]['preview']  = preview_file
            all_fonts[font_name]['proof']    = proof_file
            
            
    print 'Rendering...'
    font_names = sorted(all_fonts.keys())
    for i, f in enumerate(font_names):
        try:         
            
            name = os.path.splitext(os.path.split(f)[1])[0]
            name = name.lower()
            
            truetype_file = all_fonts[f]['truetype']
            raster_file   = all_fonts[f]['raster']
            preview_file  = all_fonts[f]['preview']
            proof_file    = all_fonts[f]['proof']

            data = render_alphabet(truetype_file, sample_size=sample_size, raster_file=raster_file)
            
            if data:
                text(words     = alphabet, 
                     font      = data,
                     height    = sample_size,
                     color     = (224, 224, 224),
                     bg_color  = (64,64,64 ),
                     bg_border = (border,border),
                     out_image = preview_file)      


            name = '%s [%s/%s] %s'%(sample_size, i+1, len(font_names), f)

            if data:
                print 'Generated: %s'%name
            else:
                print 'Failed:    %s'%name

        except:
            pass


    print ''
    print 'All Done!'



def text(words, 
         font=DEFAULT_FONT, 
         height=256, 
         out_image=None, 
         color='white', 
         bg_color=None, 
         bg_border=(0,0),
         use_kerning=True):


    if not words:
        raise Exception('Nothing to write.')



    # Load rasters
    characters = find_raster(font)
    if not characters:
        raise Exception('Bad font specified.')


    # Remove illegal chars from text
    text_ = ''
    for i, char in enumerate(words):
        if char in characters:
            text_+=char
    words = text_


    # Draw the text
    image   = Image.new('RGBA',(0,0))
    offset  = 0
    for i, char in enumerate(words):
        img     = characters[char]['image'].convert('RGBA')
        width   = characters[char]['width']

        if use_kerning and i > 0:
            offset += characters[char]['kerning'][words[i-1]]
            offset+=characters['']['width']

        img_ = Image.new('RGBA', (width,img.height), color=color)
        img_.putalpha(img.convert('L'))
        
        if isinstance(color,(list,tuple)) and len(color) == 4:
            img_ = np.array(img_)
            img_[:,:,3] = ((img_[:,:,3]/255.) * color[3]).astype('uint8')
            img_ = Image.fromarray(img_)
            
            
        image = composite(img_, image, offset=(offset,0))
        offset = image.width


    # Resize font to exact height
    w,h = image.size
    if h != height:
        ratio = height/float(h)
        w = int(w*ratio)
        image = image.resize((w,height), Image.BICUBIC)


    # Apply bg color and bg border
    if not bg_color is None:
        size = list(image.size)
        offset = [0,0]
        if not bg_border is None:
            size[0] += bg_border[0]+bg_border[2]
            size[1] += bg_border[1]+bg_border[3]
            offset[0] = bg_border[0]
            offset[1] = bg_border[1]
            
        img_ = Image.new('RGBA', size, color=bg_color)
        image = composite(image, img_, offset=offset)        
        
        
    if out_image:
        write_image(out_image, image)


    return image




#import_system_truetype(sample_size=256, full_export=False)
#import_system_truetype(sample_size=512, full_export=False)


#text(words     = 'This is an awesome test!', 
     #font      = 'arial', 
     #height    = 100, # Height of the font in pixels
     #color     = (229, 232, 232,200), # set text to light gray and a bit transparent
     #bg_color  = (66, 73, 73,50),     # set bg to dark gray but moslty transparent
     #bg_border = (50,50,50,50),       # extend bg color 50 pixels all around text (left,top,right,bottom)
     #out_image = '~/Desktop/test.png')




'''
import filecmp
rasters = glob.glob('C:/Users/ericvignola/perforce/fbavatars/library/maya/scripts/python/_common/3rd-party/rasters/lib/256/previews/*.png')
#rasters = [x for x in rasters if 'arial' in x]

results = []

for j in xrange(len(rasters)):
    duplicates = set()
    for jj in xrange(j,len(rasters)-1):
        jj+=1
        #print
        #print '%s/%s/%s'%((j+1),(jj+1),len(rasters)+1)
        #print 'R0: %s'%rasters[j]
        #print 'R1: %s'%rasters[jj]
        
        test = filecmp.cmp(rasters[j], rasters[jj], shallow=False)
        if test:
            duplicates.add(rasters[j])
            duplicates.add(rasters[jj])
            
    if duplicates:
        results.append(sorted(duplicates))
        
        print 'duplicates: %s'%duplicates
        print               
            
        #R0 = read_rasters(rasters[j])
        #R1 = read_rasters(rasters[jj])
        #count = 0
        #duplicates = set()
        
        #if set(R0) == set(R1):
            #keys = R0.keys()
            
            #for i in xrange(len(R0)):
                #k = keys[i]
                
                #if 'image' in R0[k]:
                        
                    #image0 = np.asarray(R0[k]['image'])
                    #image1 = np.asarray(R1[k]['image'])

                    #if image0.shape == image1.shape:
                        #allclose = np.allclose(image0, image1)
                        #print allclose
                        #if allclose:
                        ##if image0 == image1:
                            #duplicates.add(rasters[j])
                            #duplicates.add(rasters[jj])
        
        #if duplicates:
            #results.append(sorted(duplicates))
            
            #print 'duplicates: %s'%duplicates
            #print            
        

with open('C:/users/ericvignola/Desktop/duplicates3.pickle','w') as io:
    cPickle.dump(results,io)
        


print 'done'

'''
