from __future__ import print_function

import sys
import numpy
import cv2

import params
import util
import persp
import clean
import segment
import discard
import notes

from midiutil.MidiFile import MIDIFile


def main():
    if len(sys.argv) < 3:
        print('%s: error: invalid arguments' % sys.argv[0], file=sys.stderr)
        print('usage: %s PATH_TO_IMAGE PATH_TO_PARAMS' % sys.argv[0], file=sys.stderr)
        return 1
    else:
        filename = sys.argv[1]
        params_path = sys.argv[2]

    img = cv2.imread(filename)
    util.show(filename, img)

    filename = filename[filename.rfind("/")+1:filename.rfind(".")]
    params.load(params_path)

    # find page & correct for perspective
    img2 = persp.find_page(img)
    util.show(filename + ' (corrected)', img2, save=True)

    img3, staff_lines = clean.find_staff_lines(img2)
    util.debug('found %d staff lines: %s' % (len(staff_lines), str(staff_lines)))
    util.show(filename + ' (without staff lines)', img3, save=True)

    objs = segment.find_objects(img3)
    util.debug('found %d objects: %s' % (len(objs), str(objs)))

    objs2 = discard.filter_musical_objects(img3, objs)
    util.debug('remaining %d musical objects: %s' % (len(objs2), str(objs2)))

    notes_list = notes.find_notes(img3, objs2, staff_lines)
    util.debug('notes list: ' + str(notes_list))

    # create a MIDI file from the notes list
    midi_file = MIDIFile(numTracks=1)

    midi_file.addTrackName(0, 0, "Track 0")

    for i, (pitch, kind) in enumerate(notes_list):
        freq = midi_note_number(pitch)

        if kind == 'quarter':
            beats = 1
        elif kind == 'eighth':
            beats = 0.5
        elif kind == 'half':
            beats = 2
        elif kind == 'whole':
            beats = 4

        midi_file.addNote(
            track=0,
            channel=0,
            pitch=int(round(freq)),
            time=i,
            duration=beats,
            volume=100
        )

    out = open('output.mid', 'wb')
    midi_file.writeFile(out)
    out.close()

    cv2.waitKey(0)


def midi_note_number(s):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    if len(s) == 2:
        letter, num = s
    elif len(s) == 3:
        letter, num = s[:2], s[2]

    return (int(num) * len(NOTES)) + NOTES.index(letter)

if __name__ == '__main__':
    sys.exit(main())
