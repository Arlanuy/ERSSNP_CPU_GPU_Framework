from cusnp import SnpSystem

class PLinguaParser:

    def translate(self, string):
        """
        Convert a P Lingua string into an SN P System

        Returns
        -------
        SnpSystem instance
            The SN P System described by the P Lingua string
        """



    def convert_from_file(self, filename):
        """
        Return the SN P System described by the P Lingua file

        Returns
        -------
        SnpSystem instance
            The SN P System described by the P Lingua file
        """

        file_object = open("src/pli/"+filename)
        
        return file_object.read()

p = PLinguaParser()
print(p.convert_from_file('Doubling.pli'))

