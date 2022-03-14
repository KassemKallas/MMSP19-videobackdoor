

def parse_filename(filename):
    
    attributes = {}    

    # attacked or real?
    if ('attack_mobile' in filename) or ('attack_highdef' in filename) or ('attack_print' in filename):
        attributes['attack'] = True
        attributes['real'] = False
    else:
        attributes['attack'] = False
        attributes['real'] = True

    # parse the clip_name to find out what kind of video this is
    if (attributes['attack'] == True):               
        # attack type -- fixed or hand
        if ('hand' in filename):
            attributes['attack-type'] = 'hand'
        elif ('fixed' in filename):
            attributes['attack-type'] = 'fixed'
        
        if ('attack_mobile' in filename):
            attributes['broadcast-type'] = 'mobile' # iphone
        elif ('attack_highdef' in filename):
            attributes['broadcast-type'] = 'highdef' # ipad
        elif ('attack_print' in filename):
            attributes['broadcast-type'] = 'print' # printed

    else:
        attributes['broadcast-type'] = 'none'
        attributes['attack-type'] = 'none'
        
    # lighting condition
    lighting = 'unknown'
    if ('adverse' in filename):
        lighting = 'adverse'
    elif ('controlled' in filename):
        lighting = 'controlled'
    attributes['lighting'] = lighting
    
    return attributes
